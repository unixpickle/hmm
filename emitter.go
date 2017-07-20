package anyhmm

import "math/rand"

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
