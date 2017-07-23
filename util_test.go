package hmm

import (
	"math"
	"math/rand"
	"runtime"

	"golang.org/x/net/context"
)

func benchmarkingHMM() (*HMM, []Obs) {
	states := make([]State, 100)
	obses := make([]Obs, 100)
	for i := 0; i < 100; i++ {
		states[i] = i
		obses[i] = i
	}
	h := RandomHMM(rand.New(rand.NewSource(1337)), states, 0, obses)
	return h, []Obs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
}

func testingHMM() *HMM {
	return &HMM{
		States: []State{"A", "B", "C", "D"},
		Emitter: TabularEmitter{
			"A": map[Obs]float64{
				"x": math.Log(0.3),
				"z": math.Log(0.7),
			},
			"B": map[Obs]float64{
				"x": math.Log(0.01),
				"y": math.Log(0.69),
				"z": math.Log(0.3),
			},
			"C": map[Obs]float64{
				"x": math.Log(0.8),
				"y": math.Log(0.1),
				"z": math.Log(0.1),
			},
		},
		TerminalState: "D",
		Init: map[State]float64{
			"A": math.Log(0.4),
			"C": math.Log(0.5),
			"D": math.Log(0.1),
		},
		Transitions: map[Transition]float64{
			Transition{From: "A", To: "A"}: math.Log(0.3),
			Transition{From: "A", To: "B"}: math.Log(0.2),
			Transition{From: "A", To: "C"}: math.Log(0.5),

			Transition{From: "B", To: "A"}: math.Log(0.5),
			Transition{From: "B", To: "B"}: math.Log(0.1),
			Transition{From: "B", To: "C"}: math.Log(0.2),
			Transition{From: "B", To: "D"}: math.Log(0.2),

			Transition{From: "C", To: "A"}: math.Log(0.3),
			Transition{From: "C", To: "B"}: math.Log(0.1),
			Transition{From: "C", To: "C"}: math.Log(0.1),
			Transition{From: "C", To: "D"}: math.Log(0.5),
		},
	}
}

func obsSeqsEqual(s1, s2 []Obs) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, x := range s1 {
		if x != s2[i] {
			return false
		}
	}
	return true
}

func stateSeqsEqual(s1, s2 []State) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, x := range s1 {
		if x != s2[i] {
			return false
		}
	}
	return true
}

func sampleConditionalHidden(ctx context.Context, h *HMM, out []Obs) <-chan []State {
	res := make(chan []State, runtime.GOMAXPROCS(0))
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		go func() {
			gen := rand.New(rand.NewSource(rand.Int63()))
			for {
				states, outs := h.Sample(gen)
				if obsSeqsEqual(out, outs) {
					select {
					case res <- states:
					case <-ctx.Done():
						return
					}
				}
			}
		}()
	}
	return res
}
