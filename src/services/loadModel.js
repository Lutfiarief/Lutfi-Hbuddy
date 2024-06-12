const tf = require('@tensorflow/tfjs-node');

class Normalization extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.mean = config.mean || 0;
      this.variance = config.variance || 1;
    }
  
    computeOutputShape(inputShape) {
      return inputShape;
    }
  
    call(inputs, kwargs) {
      const meanTensor = tf.scalar(this.mean);
      const varianceTensor = tf.scalar(this.variance);
      return tf.div(tf.sub(inputs, meanTensor), tf.sqrt(varianceTensor));
    }
  
    static get className() {
      return 'Normalization';
    }
  }
  
  tf.serialization.registerClass(Normalization);

async function loadModel() {
    try {
      const model = await tf.loadLayersModel(process.env.MODEL_URL);
      return model;
    } catch (error) {
      throw new Error(`Failed to load the model: ${error.message}`);
    }
  }
  
  module.exports = loadModel;
