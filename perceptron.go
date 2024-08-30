package perceptron

import (
	"math"
	"math/rand"
)

type Perceptron struct {
	weights []float64
	bias float64
	learningRate float64
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func New(weightsAmount int) *Perceptron {
	weights := make([]float64, weightsAmount)
	for i := 0;i < weightsAmount; i++ {
		weights[i] = rand.Float64()*2 - 1
	}
	return &Perceptron{
		bias: rand.Float64()*2 - 1,
		learningRate: .1,
		weights: weights,
	}
}

func (p *Perceptron) train(input []float64, expected float64) {
	output := p.feed(input)

	err := (output - expected)
	for i := range p.weights {
		p.weights[i] -= err * input[i] * p.learningRate
	}
	p.bias -= err * p.learningRate
}

func (p *Perceptron) feed(input []float64) float64 {
	var sum float64 = 0
	for i,inputValue := range input {
		sum += inputValue * p.weights[i]
	}
	sum += p.bias
	return sigmoid(sum)
}