package hmm

import (
	"math"
	"runtime"
	"sync"
)

// BaumWelch applies a step of the BaumWelch algorithm to
// the discrete-observation HMM.
// It returns a new *HMM with some modified fields which
// reflect the updated parameters.
//
// The parallelism argument specifies the number of
// samples to process concurrently.
// If it is 0, then GOMAXPROCS is used.
//
// The HMM must use a TabularEmitter.
func BaumWelch(h *HMM, data <-chan []Obs, parallelism int) *HMM {
	if parallelism == 0 {
		parallelism = runtime.GOMAXPROCS(0)
	}
	bw := newBaumWelch(h)
	var wg sync.WaitGroup
	for i := 0; i < parallelism; i++ {
		wg.Add(1)
		go func() {
			for sample := range data {
				bw.Accumulate(sample)
			}
			wg.Done()
		}()
	}
	wg.Wait()
	bw.Normalize()
	return bw.Result()
}

type baumWelch struct {
	HMM *HMM

	InitTally map[State]float64
	InitTotal float64

	TransTally      map[Transition]float64
	FromStateTotals map[State]float64

	EmitTally  TabularEmitter
	EmitTotals map[State]float64

	UpdateLock sync.Mutex
}

func newBaumWelch(h *HMM) *baumWelch {
	return &baumWelch{
		HMM: h,

		InitTally: map[State]float64{},
		InitTotal: math.Inf(-1),

		TransTally:      map[Transition]float64{},
		FromStateTotals: map[State]float64{},

		EmitTally:  TabularEmitter{},
		EmitTotals: map[State]float64{},
	}
}

func (b *baumWelch) Accumulate(sample []Obs) {
	if len(sample) == 0 {
		if b.HMM.TerminalState != nil {
			b.UpdateLock.Lock()
			b.InitTotal = addLogs(b.InitTotal, 0)
			addToState(b.InitTally, b.HMM.TerminalState, 0)
			b.UpdateLock.Unlock()
		}
		return
	}

	fb := NewForwardBackward(b.HMM, sample)
	initDist := fb.Dist(0)

	b.UpdateLock.Lock()
	b.InitTotal = addLogs(b.InitTotal, 0)
	for state, prob := range initDist {
		addToState(b.InitTally, state, prob)
	}
	b.UpdateLock.Unlock()

	for t := 1; t <= len(sample); t++ {
		if t == len(sample) && b.HMM.TerminalState == nil {
			// Looking at the final state transitions don't make
			// much sense, since we really have no idea what the
			// next state should be.
			break
		}

		prevDist := fb.Dist(t - 1)
		condDist := fb.CondDist(t)

		b.UpdateLock.Lock()
		for state, prob := range prevDist {
			addToState(b.FromStateTotals, state, prob)
		}
		for from, tos := range condDist {
			for to, condProb := range tos {
				joint := condProb + prevDist[from]
				trans := Transition{From: from, To: to}
				if oldProb, ok := b.TransTally[trans]; ok {
					b.TransTally[trans] = addLogs(oldProb, joint)
				} else {
					b.TransTally[trans] = joint
				}
			}
		}
		b.UpdateLock.Unlock()
	}

	for t, obs := range sample {
		for state, prob := range fb.Dist(t) {
			b.UpdateLock.Lock()
			if _, ok := b.EmitTally[state]; !ok {
				b.EmitTally[state] = map[Obs]float64{}
			}
			addToState(b.EmitTotals, state, prob)
			emissions := b.EmitTally[state]
			if oldProb, ok := emissions[obs]; ok {
				emissions[obs] = addLogs(oldProb, prob)
			} else {
				emissions[obs] = prob
			}
			b.UpdateLock.Unlock()
		}
	}
}

func (b *baumWelch) Normalize() {
	for trans := range b.TransTally {
		b.TransTally[trans] -= b.FromStateTotals[trans.From]
	}
	for state := range b.InitTally {
		b.InitTally[state] -= b.InitTotal
	}
	for state, emissions := range b.EmitTally {
		total := b.EmitTotals[state]
		for obs := range emissions {
			emissions[obs] -= total
		}
	}
}

func (b *baumWelch) Result() *HMM {
	res := *b.HMM
	res.Emitter = b.EmitTally
	res.Transitions = b.TransTally
	res.Init = b.InitTally
	return &res
}
