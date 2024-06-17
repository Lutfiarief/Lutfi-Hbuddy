'use strict';

const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
require('dotenv').config();
const log = console.log;

// Import the custom layer
require('./normalization');

// Import preprocessInput function from tfjs
const { preprocessInput } = require('./efficientnet_preprocess');

let model;

// Define the plant information
const plantInfo = {
    'Asoka': {
        binomial: "Saraca Asoca",
        description: "Tumbuhan ini dikenal sebagai simbol keindahan dan kedamaian dalam banyak budaya di seluruh dunia...",
        benefit: ["1. Meringankan nyeri haid", "2. Menjaga kesehatan kulit", "3. Obat anti radang"]
    },
    'Bunga Telang': {
        binomial: "Clitoria Ternatea",
        description: "Bunga Telang atau Clitoria ternatea, umumnya dikenal dengan 'butterfly pea'...",
        benefit: ["1. Menurunkan demam dan meredakan rasa nyeri", "2. Meredakan gejala alergi", "3. Melancarkan aliran darah ke kapiler mata"]
    },
    'Daun Jambu Biji': {
        binomial: "Psidium guajava",
        description: "Daun jambu biji (Psidium guajava) adalah bagian dari pohon jambu biji yang tumbuh di daerah tropis...",
        benefit: ["1. Meningkatkan kekebalan tubuh", "2. Menjaga kadar gula darah tetap stabil", "3. Menjaga kesehatan sistem pencernaan"]
    },
    'Daun Jarak': {
        binomial: "Ricinus Communis",
        description: "Daun jarak (Jatropha curcas) adalah tanaman herbal yang tumbuh subur di daerah tropis...",
        benefit: ["1. Mengatasi sembelit", "2. Mengunci kelembapan kulit", "3. Mempercepat penyembuhan luka"]
    },
    'Daun Jeruk Nipis': {
        binomial: "Citrus Aurantifolia",
        description: "Daun jeruk nipis (Citrus aurantiifolia) adalah tanaman herbal yang sering digunakan dalam berbagai pengobatan tradisional...",
        benefit: ["1. Mempercepat penyembuhan luka...", "2. Meningkatkan kesehatan kulit...", "3. Mengatasi masalah pencernaan..."]
    },
    'Kayu Putih': {
        binomial: "Melaleuca Leucadendra",
        description: "Daun kayu putih (Melaleuca alternifolia) merupakan bagian dari pohon kayu putih yang tumbuh di Australia...",
        benefit: ["1. Pereda sakit...", "2. Respon imun...", "3. Kondisi pernafasan..."]
    },
    'Daun Pepaya': {
        binomial: "Carica Pepaya",
        description: "Daun pepaya (Carica papaya) adalah bagian dari tanaman pepaya yang memiliki berbagai manfaat kesehatan...",
        benefit: ["1. Meningkatkan pencernaan...", "2. Anti-inflamasi...", "3. Menurunkan tekanan darah..."]
    },
    'Sirih': {
        binomial: "Piper betle",
        description: "Daun sirih (Piper betle) adalah bagian dari tanaman sirih yang tumbuh subur di berbagai daerah tropis...",
        benefit: ["1. Menyehatkan saluran pencernaan...", "2. Mengatasi sembelit...", "3. Menjaga kesehatan mulut dan gigi..."]
    },
    'Lidah Buaya': {
        binomial: "Aloe Vera",
        description: "Lidah buaya (Aloe vera), adalah spesies tanaman dengan daun berdaging tebal dari genus Aloe...",
        benefit: ["1. Membantu mengatasi permasalahan kulit...", "2. Memelihara kesehatan kulit...", "3. Membantu menutrisi rambut..."]
    },
    'Semanggi': {
        binomial: "Marsilea Crenata Presl",
        description: "Daun semanggi adalah bagian dari tanaman semanggi (Oxalis spp.) yang tumbuh secara alami di berbagai belahan dunia...",
        benefit: ["1. Antioksidan yang kuat...", "2. Meningkatkan kekuatan tulang...", "3. Mendukung kesehatan prostat..."]
    }
};

// Define class names
const class_names = ['Asoka', 'Bunga Telang', 'Daun Jambu Biji', 'Daun Jarak', 'Daun Jeruk Nipis', 'Kayu Putih', 'Daun Pepaya', 'Sirih', 'Lidah Buaya', 'Semanggi'];

// Load the model
const loadModel = async () => {
    if (!model) {
        const modelUrl = process.env.MODEL_URL;
        log(`Loading model from ${modelUrl}`);
        model = await tf.loadLayersModel(modelUrl);
        log('Model loaded successfully');
    }
};

const classifyImage = async (request, h) => {
    try {
        log('Received request for image classification');
        const { file } = request.payload;

        if (!file) {
            log('No file found in the request payload');
            return h.response({ error: 'No file uploaded' }).code(400);
        }

        log(`Received file: ${file.hapi.filename}`);
        const uploadPath = path.join(__dirname, '../uploads', file.hapi.filename);

        const fileStream = fs.createWriteStream(uploadPath);
        await new Promise((resolve, reject) => {
            file.pipe(fileStream);
            file.on('end', resolve);
            file.on('error', reject);
        });

        log(`File saved to ${uploadPath}`);

        const image = fs.readFileSync(uploadPath);
        const tensor = tf.node.decodeImage(image, 3)
            .resizeBilinear([224, 224])
            .expandDims()
            .toFloat();

        // Apply the EfficientNet-specific preprocessing
        const preprocessedTensor = preprocessInput(tensor);

        log('Image processed into tensor');

        await loadModel();
        const predictions = await model.predict(preprocessedTensor).data();
        log('Model prediction completed');

        // Get the index of the highest prediction
        const maxPredictionIndex = predictions.indexOf(Math.max(...predictions));
        const predictedPlant = Object.keys(plantInfo)[maxPredictionIndex];

        // Validate prediction with if statement
        let label, binomial, description, benefit;

        if (predictedPlant === 'Asoka') {
            label = 'Asoka';
            ({ binomial, description, benefit } = plantInfo['Asoka']);
        } else if (predictedPlant === 'Bunga Telang') {
            label = 'Bunga Telang';
            ({ binomial, description, benefit } = plantInfo['Bunga Telang']);
        } else if (predictedPlant === 'Daun Jambu Biji') {
            label = 'Daun Jambu Biji';
            ({ binomial, description, benefit } = plantInfo['Daun Jambu Biji']);
        } else if (predictedPlant === 'Daun Jarak') {
            label = 'Daun Jarak';
            ({ binomial, description, benefit } = plantInfo['Daun Jarak']);
        } else if (predictedPlant === 'Daun Jeruk Nipis') {
            label = 'Daun Jeruk Nipis';
            ({ binomial, description, benefit } = plantInfo['Daun Jeruk Nipis']);
        } else if (predictedPlant === 'Kayu Putih') {
            label = 'Kayu Putih';
            ({ binomial, description, benefit } = plantInfo['Kayu Putih']);
        } else if (predictedPlant === 'Daun Pepaya') {
            label = 'Daun Pepaya';
            ({ binomial, description, benefit } = plantInfo['Daun Pepaya']);
        } else if (predictedPlant === 'Sirih') {
            label = 'Sirih';
            ({ binomial, description, benefit } = plantInfo['Sirih']);
        } else if (predictedPlant === 'Lidah Buaya') {
            label = 'Lidah Buaya';
            ({ binomial, description, benefit } = plantInfo['Lidah Buaya']);
        } else if (predictedPlant === 'Semanggi') {
            label = 'Semanggi';
            ({ binomial, description, benefit } = plantInfo['Semanggi']);
        } else {
            throw new Error('Prediction did not match any known plant');
        }

        log(`Predicted plant: ${label}`);

        return h.response({
            plant: label,
            binomial,
            description,
            benefit
        }).code(200);
    } catch (error) {
        console.error('Error during image classification:', error);
        return h.response({ error: 'Failed to process image' }).code(500);
    }
};

module.exports = {
    classifyImage
};
