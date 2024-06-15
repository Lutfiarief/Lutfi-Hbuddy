const tf = require('@tensorflow/tfjs-node');

class Normalization extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.axis = config.axis || -1;
    }

    call(inputs) {
        const input = inputs[0];
        const mean = input.mean(this.axis, true);
        const variance = input.sub(mean).square().mean(this.axis, true);
        const epsilon = tf.scalar(1e-7);
        return input.sub(mean).div(variance.add(epsilon).sqrt());
    }

    static get className() {
        return 'Normalization';
    }
}

// Register the Normalization class
tf.serialization.registerClass(Normalization);

module.exports = Normalization;
