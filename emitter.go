package hmm

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer(TabularEmitter{}.SerializerType(),
		DeserializeTabularEmitter)
}

// An Obs is an observation for a single timestep.
type Obs interface{}

// An Emitter performs operations on the conditional
// distribution of observations.
type Emitter interface {
	// Sample samples an observation, conditioned on the
	// hidden state.
	//
	// If gen is not nil, its use is optional but may
	// improve performance in concurrent applications.
	Sample(gen *rand.Rand, state State) Obs

	// LogProbs returns the conditional log probability of
	// the observation conditioned on each of the states.
	LogProbs(obs Obs, states ...State) []float64
}

// A TabularEmitter is an Emitter which uses a table of
// pre-determined probabilities and observations.
//
// Each State is mapped to a mapping from observations
// to their corresponding log probabilities.
// Absent observations have a probability of 0.
type TabularEmitter map[State]map[Obs]float64

// DeserializeTabularEmitter deserializes a TabularEmitter.
func DeserializeTabularEmitter(d []byte) (t TabularEmitter, err error) {
	defer essentials.AddCtxTo("deserialize TabularEmitter", &err)
	var states []serializer.Serializer
	var obses []serializer.Serializer
	var probs []float64
	if err := serializer.DeserializeAny(d, &states, &obses, &probs); err != nil {
		return nil, err
	}
	if len(states) != len(obses) || len(probs) != len(obses) {
		return nil, errors.New("mismatching slice lengths")
	} else if !serializersComparable(states, obses) {
		return nil, errors.New("State or Obs not comparable")
	}
	t = TabularEmitter{}
	for i, state := range states {
		if _, ok := t[state]; !ok {
			t[state] = map[Obs]float64{}
		}
		t[state][obses[i]] = probs[i]
	}
	return t, nil
}

// Sample samples an observation from the state.
func (t TabularEmitter) Sample(gen *rand.Rand, state State) Obs {
	if len(t[state]) == 0 {
		panic("no entries for the given state")
	}
	var obses []Obs
	var probs []float64
	for obs, prob := range t[state] {
		obses = append(obses, obs)
		probs = append(probs, math.Exp(prob))
	}
	return obses[sampleIndex(gen, probs)]
}

// LogProbs computes the conditional probabilities.
func (t TabularEmitter) LogProbs(obs Obs, states ...State) []float64 {
	var res []float64
	for _, state := range states {
		if prob, ok := t[state][obs]; ok {
			res = append(res, prob)
		} else {
			res = append(res, math.Inf(-1))
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a TabularEmitter with the serializer package.
func (t TabularEmitter) SerializerType() string {
	return "github.com/unixpickle/hmm.TabularEmitter"
}

// Serialize serializes the TabularEmitter.
//
// For this to work, the states and observations must
// implement serializer.Serializer.
func (t TabularEmitter) Serialize() (data []byte, err error) {
	defer essentials.AddCtxTo("serialize TabularEmitter", &err)
	var states []serializer.Serializer
	var obses []serializer.Serializer
	var probs []float64
	for state, dist := range t {
		stateSer, ok := state.(serializer.Serializer)
		if !ok {
			return nil, fmt.Errorf("not a Serializer: %T", state)
		}
		for obs, prob := range dist {
			obsSer, ok := obs.(serializer.Serializer)
			if !ok {
				return nil, fmt.Errorf("not a Serializer: %T", obs)
			}
			states = append(states, stateSer)
			obses = append(obses, obsSer)
			probs = append(probs, prob)
		}
	}
	return serializer.SerializeAny(states, obses, probs)
}
