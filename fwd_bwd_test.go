package hmm

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/approb"

	"golang.org/x/net/context"
)

const (
	TestSamples   = 2000000
	TestThreshold = 0.001
)

func TestForwardProbs(t *testing.T) {
	h := testingHMM()
	out := []Obs{"x", "z", "y", "x"}

	expected := approxForwardProbs(h, out)
	actual := actualForwardProbs(h, out)

	if len(actual) != len(expected) {
		t.Fatal("length mismatch")
	}
	for i, act := range actual {
		exp := expected[i]
		for _, state := range h.States {
			a := act[state]
			x := exp[state]
			if (math.IsNaN(a) != math.IsNaN(x)) || math.Abs(a-x) > TestThreshold {
				t.Errorf("time %d state %v: expected %v but got %v",
					i, state, x, a)
			}
		}
	}
}

func approxForwardProbs(h *HMM, out []Obs) []map[State]float64 {
	list := make([]map[State]float64, len(out))
	for i := range list {
		list[i] = map[State]float64{}
	}
	for i := 0; i < TestSamples; i++ {
		sampleState, sampleOut := h.Sample(nil)
		for j := 0; j < len(out) && j < len(sampleOut); j++ {
			if obsSeqsEqual(sampleOut[:j+1], out[:j+1]) {
				list[j][sampleState[j]] += 1.0 / TestSamples
			}
		}
	}
	return list
}

func actualForwardProbs(h *HMM, out []Obs) []map[State]float64 {
	var res []map[State]float64
	for obj := range ForwardProbs(h, out) {
		for k, v := range obj {
			obj[k] = math.Exp(v)
		}
		res = append(res, obj)
	}
	return res
}

func TestBackwardProbs(t *testing.T) {
	h := testingHMM()
	out := []Obs{"x", "z", "y", "x"}

	expected := approxBackwardProbs(h, out)
	actual := actualBackwardProbs(h, out)

	if len(actual) != len(expected) {
		t.Fatal("length mismatch")
	}
	for i, act := range actual {
		exp := expected[i]
		for _, state := range h.States {
			a := act[state]
			x := exp[state]
			if (math.IsNaN(a) != math.IsNaN(x)) || math.Abs(a-x) > TestThreshold {
				t.Errorf("time %d state %v: expected %v but got %v",
					i, state, x, a)
			}
		}
	}
}

func approxBackwardProbs(h *HMM, out []Obs) []map[State]float64 {
	matches := make([]map[State]float64, len(out))
	totals := map[State]float64{}
	for i := range matches {
		matches[i] = map[State]float64{}
	}
	for i := 0; i < TestSamples; i++ {
		sampleState, sampleOut := h.Sample(nil)
		for j := range sampleOut {
			state := sampleState[len(sampleState)-(j+1)]
			totals[state]++
			if j >= len(out) {
				continue
			}
			out1 := sampleOut[len(sampleOut)-j:]
			out2 := out[len(out)-j:]
			if obsSeqsEqual(out1, out2) {
				matches[j][state]++
			}
		}
	}
	for _, subMatches := range matches {
		for state := range subMatches {
			subMatches[state] /= totals[state]
		}
	}
	return matches
}

func actualBackwardProbs(h *HMM, out []Obs) []map[State]float64 {
	var res []map[State]float64
	for obj := range BackwardProbs(h, out) {
		for k, v := range obj {
			obj[k] = math.Exp(v)
		}
		res = append(res, obj)
	}
	return res
}

func TestSmoother(t *testing.T) {
	h := testingHMM()
	out := []Obs{"x", "z", "y", "x"}

	ctx, cancel := context.WithCancel(context.Background())
	var samples [][]State
	ch := sampleConditionalHidden(ctx, h, out)
	for i := 0; i < 3000; i++ {
		samples = append(samples, <-ch)
	}
	cancel()

	stateToNum := map[State]float64{"A": 0, "B": 1, "C": 2, "D": 3}
	smoother := NewSmoother(h, out)
	for idx := range out {
		t.Run(fmt.Sprintf("Time%d", idx), func(t *testing.T) {
			dist := smoother.Dist(idx)
			var states []State
			var probs []float64
			for state, prob := range dist {
				states = append(states, state)
				probs = append(probs, math.Exp(prob))
			}
			corr := approb.Correlation(15000, 0.5, func() float64 {
				sample := samples[rand.Intn(len(samples))]
				return stateToNum[sample[idx]]
			}, func() float64 {
				return stateToNum[states[sampleIndex(nil, probs)]]
			})
			if corr < 0.999 {
				t.Errorf("correlation is %f (expected 1)", corr)
			}
		})
	}
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