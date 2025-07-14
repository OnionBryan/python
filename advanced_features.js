class V24NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize, opts={}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = opts.learningRate || 0.01;
    this.momentum = opts.momentum || 0.9;
    this.lossType = opts.lossType || 'mse';
    this.weightsIH = math.random([hiddenSize, inputSize], -1, 1);
    this.weightsHO = math.random([outputSize, hiddenSize], -1, 1);
    this.biasH = math.zeros(hiddenSize);
    this.biasO = math.zeros(outputSize);
    this.prevDeltaIH = math.zeros([hiddenSize, inputSize]);
    this.prevDeltaHO = math.zeros([outputSize, hiddenSize]);
    this.epochLosses = [];
    this.trainCount = 0;
  }
  sigmoid(x){return 1/(1+Math.exp(-Math.max(-500,Math.min(500,x))))}
  sigmoidDerivative(x){return x*(1-x)}
  forward(inputs, weights, bias){
    const wx = math.add(math.multiply(weights, inputs), bias);
    return wx.map(this.sigmoid);
  }
  train(inputs, targets){
    const hidden = this.forward(inputs, this.weightsIH, this.biasH);
    const outputs = this.forward(hidden, this.weightsHO, this.biasO);
    const outputErrors = math.subtract(targets, outputs);
    const hiddenErrors = math.multiply(math.transpose(this.weightsHO), outputErrors);
    this.updateWeights(this.weightsHO, this.prevDeltaHO, hidden, outputErrors, outputs);
    this.updateWeights(this.weightsIH, this.prevDeltaIH, inputs, hiddenErrors, hidden);
    this.biasO = math.add(this.biasO, math.multiply(this.learningRate, outputErrors));
    this.biasH = math.add(this.biasH, math.multiply(this.learningRate, hiddenErrors));
    const loss = math.mean(math.square(outputErrors));
    this.epochLosses.push(loss);
    this.trainCount++;
  }
  updateWeights(weights, prevDelta, inputs, errors, outputs){
    const grad = math.dotMultiply(errors, outputs.map(this.sigmoidDerivative));
    const delta = math.add(
      math.multiply(this.learningRate, math.multiply(math.reshape(grad, [grad.length,1]), [inputs])),
      math.multiply(this.momentum, prevDelta)
    );
    for(let r=0;r<weights.length;r++){
      for(let c=0;c<weights[r].length;c++){
        weights[r][c]+=delta[r][c];
      }
    }
    return delta;
  }
  predict(inputs){
    const hidden = this.forward(inputs, this.weightsIH, this.biasH);
    return this.forward(hidden, this.weightsHO, this.biasO);
  }
  getAccuracy(){
    if(this.epochLosses.length===0) return 0;
    const recent=this.epochLosses.slice(-10);
    const meanLoss=recent.reduce((s,l)=>s+l,0)/recent.length;
    return Math.max(0,1-meanLoss);
  }
}

class XORIntelligenceSystem {
  constructor(nodeCount){
    this.nodeCount = nodeCount;
    this.network = new V24NeuralNetwork(4,8,1,{learningRate:0.02});
  }
  learnPatterns(patterns){
    patterns.forEach(p=>{
      const input=[p.nodeA/this.nodeCount,p.nodeB/this.nodeCount,p.compatibility,p.xorResult];
      this.network.train(input,[p.xorResult]);
    });
  }
  predictXOR(a,b,c){
    const input=[a/this.nodeCount,b/this.nodeCount,c,0];
    return this.network.predict(input)[0];
  }
  getAccuracy(){return this.network.getAccuracy();}
}

class QuantumBitSystem {
  constructor(nodeCount){
    this.nodeCount=nodeCount;
    this.qubits=Array.from({length:nodeCount},()=>({state:0.5,phase:0}));
    this.entanglementMatrix=math.zeros([nodeCount,nodeCount]);
  }
  setQubitState(i,state){
    if(i<this.nodeCount){
      this.qubits[i].state=Math.max(0,Math.min(1,state));
      this.qubits[i].phase=Math.random()*2*Math.PI;
    }
  }
  entangleQubits(a,b,strength){
    if(a<this.nodeCount&&b<this.nodeCount){
      this.entanglementMatrix.subset(math.index(a,b),strength);
      this.entanglementMatrix.subset(math.index(b,a),strength);
    }
  }
  measureQubit(i){
    if(i>=this.nodeCount) return 0;
    return Math.random()<this.qubits[i].state?1:0;
  }
  getSystemEntanglement(){
    const total=math.sum(this.entanglementMatrix)-math.trace(this.entanglementMatrix);
    return total/(this.nodeCount*(this.nodeCount-1));
  }
}
