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
	HMM           *HMM
	TerminalIndex int

	InitTally *fastStateMap
	InitTotal float64

	TransTally      []*fastStateMap
	FromStateTotals *fastStateMap

	EmitTally  []map[Obs]float64
	EmitTotals *fastStateMap

	UpdateLock sync.Mutex
}

func newBaumWelch(h *HMM) *baumWelch {
	b := &baumWelch{
		HMM: h,

		InitTally: newFastStateMap(h),
		InitTotal: math.Inf(-1),

		TransTally:      make([]*fastStateMap, len(h.States)),
		FromStateTotals: newFastStateMap(h),

		EmitTally:  make([]map[Obs]float64, len(h.States)),
		EmitTotals: newFastStateMap(h),
	}
	for i, state := range h.States {
		b.TransTally[i] = newFastStateMap(h)
		b.EmitTally[i] = map[Obs]float64{}
		if state == h.TerminalState {
			b.TerminalIndex = i
		}
	}
	return b
}

func (b *baumWelch) Accumulate(sample []Obs) {
	if len(sample) == 0 {
		if b.HMM.TerminalState != nil {
			b.UpdateLock.Lock()
			b.InitTotal = addLogs(b.InitTotal, 0)
			b.InitTally.AddLog(b.TerminalIndex, 0)
			b.UpdateLock.Unlock()
		}
		return
	}

	fb := NewForwardBackward(b.HMM, sample)

	var dists []*fastStateMap
	for t := 0; t < len(sample); t++ {
		dists = append(dists, newFastStateMapFrom(b.HMM, fb.Dist(t)))
	}

	var condDists [][]*fastStateMap
	for t := 1; t <= len(sample); t++ {
		if t == len(sample) && b.HMM.TerminalState == nil {
			// Looking at the final state transitions don't make
			// much sense, since we really have no idea what the
			// next state should be.
			break
		}
		condDists = append(condDists, fb.fastCondDist(t))
	}

	b.UpdateLock.Lock()
	b.InitTotal = addLogs(b.InitTotal, 0)
	dists[0].Iter(func(state int, val float64) {
		b.InitTally.AddLog(state, val)
	})
	b.UpdateLock.Unlock()

	for i, condDist := range condDists {
		prevDist := dists[i]
		prevDist.Iter(func(from int, prevProb float64) {
			b.UpdateLock.Lock()
			b.FromStateTotals.AddLog(from, prevProb)
			condDist[from].Iter(func(to int, condProb float64) {
				joint := condProb + prevProb
				b.TransTally[from].AddLog(to, joint)
			})
			b.UpdateLock.Unlock()
		})
	}

	for t, obs := range sample {
		b.UpdateLock.Lock()
		dists[t].Iter(func(state int, prob float64) {
			b.EmitTotals.AddLog(state, prob)
			emissions := b.EmitTally[state]
			if oldProb, ok := emissions[obs]; ok {
				emissions[obs] = addLogs(oldProb, prob)
			} else {
				emissions[obs] = prob
			}
		})
		b.UpdateLock.Unlock()
	}
}

func (b *baumWelch) Normalize() {
	b.FromStateTotals.Iter(func(from int, total float64) {
		b.TransTally[from].AddAll(-total)
	})
	b.InitTally.AddAll(-b.InitTotal)
	b.EmitTotals.Iter(func(state int, total float64) {
		tally := b.EmitTally[state]
		for obs := range tally {
			tally[obs] -= total
		}
	})
}

func (b *baumWelch) Result() *HMM {
	res := *b.HMM

	res.Init = b.InitTally.Map()

	te := TabularEmitter{}
	for stateIdx, obses := range b.EmitTally {
		te[b.HMM.States[stateIdx]] = obses
	}
	res.Emitter = te

	res.Transitions = map[Transition]float64{}
	for from, tos := range b.TransTally {
		fromState := b.HMM.States[from]
		tos.Iter(func(to int, prob float64) {
			t := Transition{
				From: fromState,
				To:   b.HMM.States[to],
			}
			res.Transitions[t] = prob
		})
	}

	return &res
}
