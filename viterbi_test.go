package hmm

import (
	"testing"

	"golang.org/x/net/context"
)

func TestMostLikelyTerminal(t *testing.T) {
	h := testingHMM()
	obs := []Obs{"x", "z", "y", "x"}
	actual := MostLikely(h, obs)
	expected := approxMostLikelyTerminal(h, obs)
	if !stateSeqsEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestMostLikelyNonTerminal(t *testing.T) {
	h := testingHMM()
	obs := []Obs{"x", "z", "x", "x"}

	h.TerminalState = nil
	h.Emitter.(TabularEmitter)["D"] = map[Obs]float64{
		"x": 0,
	}
	h.Transitions[Transition{From: "D", To: "D"}] = 0

	actual := MostLikely(h, obs)
	expected := approxMostLikelyNonTerminal(h, obs)
	if !stateSeqsEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func approxMostLikelyTerminal(h *HMM, out []Obs) []State {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch := sampleConditionalHidden(ctx, h, out)
	var seqs [][]State
	for i := 0; i < 2000; i++ {
		seqs = append(seqs, <-ch)
	}
	return mostFrequentStateSeq(seqs)
}

func approxMostLikelyNonTerminal(h *HMM, out []Obs) []State {
	var count int
	var seqs [][]State
	for count < 5000 {
		states, outs := h.SampleLen(nil, len(out))
		if obsSeqsEqual(outs, out) {
			count++
			seqs = append(seqs, states[:len(out)])
		}
	}
	return mostFrequentStateSeq(seqs)
}

func mostFrequentStateSeq(all [][]State) []State {
	counts := map[string]int{}
	for _, hidden := range all {
		hiddenStr := ""
		for _, state := range hidden {
			hiddenStr += state.(string)
		}
		counts[hiddenStr]++
	}

	var maxRes string
	var maxCount int
	for res, count := range counts {
		if count >= maxCount {
			maxCount = count
			maxRes = res
		}
	}

	var resStates []State
	for _, ch := range maxRes {
		resStates = append(resStates, string(ch))
	}
	return resStates
}
