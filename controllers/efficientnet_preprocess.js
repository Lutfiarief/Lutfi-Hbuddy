const tf = require('@tensorflow/tfjs-node');

const preprocessInput = (tensor) => {
    const offset = tf.scalar(127.5);
    const normalized = tensor.sub(offset).div(offset);
    return normalized;
};

module.exports = { preprocessInput };
