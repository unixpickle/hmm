package hmm

import "testing"

func TestBaumWelch(t *testing.T) {
	states := []State{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	obses := []Obs{"a", "b", "c", "d", "e", "f"}
	makeSamples := func() <-chan []Obs {
		res := make(chan []Obs, 2)
		res <- []Obs{"a", "b", "c"}
		res <- []Obs{"d", "e", "f"}
		close(res)
		return res
	}
	h := RandomHMM(states, 9, obses)
	logLikelihood := func() float64 {
		var product float64
		for sample := range makeSamples() {
			product += NewForwardBackward(h, sample).LogLikelihood()
		}
		return product
	}
	for i := 0; i < 3; i++ {
		oldLikelihood := logLikelihood()
		h = BaumWelch(h, makeSamples(), 0)
		newLikelihood := logLikelihood()
		if newLikelihood < oldLikelihood {
			t.Errorf("expected new likelihood (%f) to be greater than %f", newLikelihood,
				oldLikelihood)
		}
	}
}
