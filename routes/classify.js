'use strict';

const ImageController = require('../controllers/classifyImage');
const Joi = require('joi');

module.exports = [
    {
        method: 'POST',
        path: '/classify',
        options: {
            handler: ImageController.classifyImage,
            payload: {
                output: 'stream',
                parse: true,
                allow: 'multipart/form-data',
                multipart: true
            },
            validate: {
                payload: Joi.object({
                    file: Joi.any().meta({ swaggerType: 'file' }).required()
                })
            }
        }
    }
];
