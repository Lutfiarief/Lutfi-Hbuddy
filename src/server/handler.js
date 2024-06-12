const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');
const storeData = require('../services/storeData');
const InputError = require('../exceptions/InputError');

async function postPredictHandler(request, h) {
    try {
        console.log('Request received:', request.payload);

        const { image } = request.payload;
        const { model } = request.server.app;

        if (!image) {
            throw new InputError('Image is required');
        }

        // Ensure the image is a Buffer
        if (!Buffer.isBuffer(image)) {
            throw new InputError('Image must be a Buffer');
        }

        const { label, binomial, description, benefit, confidenceScore } = await predictClassification(model, image);
        const id = crypto.randomUUID();
        const createdAt = new Date().toISOString();

        const data = {
            id,
            result: label,
            binomial,
            description,
            benefit,
            confidenceScore,
            createdAt
        };

        await storeData(id, data);

        const response = h.response({
            status: 'success',
            message: 'Model is predicted successfully',
            data
        });
        response.code(201);
        return response;
    } catch (error) {
        console.error('Error occurred:', error);

        if (error instanceof InputError) {
            const response = h.response({
                status: 'fail',
                message: error.message,
            });
            response.code(400);
            return response;
        }

        const response = h.response({
            status: 'error',
            message: 'An internal server error occurred',
        });
        response.code(500);
        return response;
    }
}

module.exports = postPredictHandler;
