package hmm

import (
	"math"
	"reflect"
	"testing"

	"github.com/unixpickle/serializer"
)

func TestSerialize(t *testing.T) {
	hmm1 := serializableHMM()
	data, err := serializer.SerializeAny(hmm1)
	if err != nil {
		t.Fatal(err)
	}
	var hmm2 *HMM
	if err := serializer.DeserializeAny(data, &hmm2); err != nil {
		t.Fatal(err)
	}

	seq := []Obs{serializer.String("x"), serializer.String("y"),
		serializer.String("z")}
	latent1 := MostLikely(hmm1, seq)
	latent2 := MostLikely(hmm2, seq)
	if !reflect.DeepEqual(latent1, latent2) {
		t.Errorf("expected %v but got %v", latent1, latent2)
	}
	prob1 := LogLikelihood(hmm1, seq)
	prob2 := LogLikelihood(hmm2, seq)
	if math.Abs(prob1-prob2) > 1e-4 {
		t.Errorf("expected %f but got %v", prob1, prob2)
	}
}

func serializableHMM() *HMM {
	h := testingHMM()

	for i, state := range h.States {
		h.States[i] = serializer.String(state.(string))
	}

	h.TerminalState = serializer.String(h.TerminalState.(string))

	newInit := map[State]float64{}
	for state, prob := range h.Init {
		newInit[serializer.String(state.(string))] = prob
	}
	h.Init = newInit

	newTrans := map[Transition]float64{}
	for trans, prob := range h.Transitions {
		t := Transition{
			From: serializer.String(trans.From.(string)),
			To:   serializer.String(trans.To.(string)),
		}
		newTrans[t] = prob
	}
	h.Transitions = newTrans

	emitter := TabularEmitter{}
	for state, dist := range h.Emitter.(TabularEmitter) {
		newDist := map[Obs]float64{}
		for obs, prob := range dist {
			newDist[serializer.String(obs.(string))] = prob
		}
		emitter[serializer.String(state.(string))] = newDist
	}
	h.Emitter = emitter

	return h
}
