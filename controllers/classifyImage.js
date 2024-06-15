'use strict';

const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
require('dotenv').config();
const log = console.log;

// Import the custom layer
require('./normalization');

let model;

// Define the plant information
const plantInfo = {
    'Asoka': {
        binomial: "Saraca Asoca",
        description: "Tumbuhan ini dikenal sebagai simbol keindahan dan kedamaian dalam banyak budaya di seluruh dunia...",
        benefit:["1. Meringankan nyeri haid",
                 "2. Menjaga kesehatan kulit",
                 "3. Obat anti radang"] 
    },
    'Bunga Telang': {
        binomial: "Clitoria Ternatea",
        description: "Bunga Telang atau Clitoria ternatea, umumnya dikenal dengan 'butterfly pea'...",
        benefit:["1. Menurunkan demam dan meredakan rasa nyeri",
                 "2. Meredakan gejala alergi",
                 "3. Melancarkan aliran darah ke kapiler mata"] 
    },
    'Daun Jambu Biji': {
        binomial: "Psidium guajava",
        description: "Daun jambu biji (Psidium guajava) adalah bagian dari pohon jambu biji yang tumbuh di daerah tropis...",
        benefit:["1. Meningkatkan kekebalan tubuh",
                 "2. Menjaga kadar gula darah tetap stabil",
                 "3. Menjaga kesehatan sistem pencernaan"]
    },
    'Daun Jarak': {
        binomial: "Ricinus Communis",
        description: "Daun jarak (Jatropha curcas) adalah tanaman herbal yang tumbuh subur di daerah tropis...",
        benefit:["1. Mengatasi sembelit",
                 "2. Mengunci kelembapan kulit",
                 "3. Mempercepat penyembuhan luka"] 
    },
    'Daun Jeruk Nipis': {
        binomial: "Citrus Aurantifolia",
        description: "Daun jeruk nipis (Citrus aurantiifolia) adalah tanaman herbal yang sering digunakan dalam berbagai pengobatan tradisional...",
        benefit: "1. Mempercepat penyembuhan luka...\n2. Meningkatkan kesehatan kulit...\n3. Mengatasi masalah pencernaan..."
    },
    'Daun Pepaya': {
        binomial: "Carica Pepaya",
        description: "Daun pepaya (Carica papaya) adalah bagian dari tanaman pepaya yang memiliki berbagai manfaat kesehatan...",
        benefit: "1. Meningkatkan pencernaan...\n2. Anti-inflamasi...\n3. Menurunkan tekanan darah..."
    },
    'Kayu Putih': {
        binomial: "Melaleuca Leucadendra",
        description: "Daun kayu putih (Melaleuca alternifolia) merupakan bagian dari pohon kayu putih yang tumbuh di Australia...",
        benefit: "1. Pereda sakit...\n2. Respon imun...\n3. Kondisi pernafasan..."
    },
    'Lidah Buaya': {
        binomial: "Aloe Vera",
        description: "Lidah buaya (Aloe vera), adalah spesies tanaman dengan daun berdaging tebal dari genus Aloe...",
        benefit: "1. Membantu mengatasi permasalahan kulit...\n2. Memelihara kesehatan kulit...\n3. Membantu menutrisi rambut..."
    },
    'Semanggi': {
        binomial: "Marsilea Crenata Presl",
        description: "Daun semanggi adalah bagian dari tanaman semanggi (Oxalis spp.) yang tumbuh secara alami di berbagai belahan dunia...",
        benefit: "1. Antioksidan yang kuat...\n2. Meningkatkan kekuatan tulang...\n3. Mendukung kesehatan prostat..."
    },
    'Sirih': {
        binomial: "Piper betle",
        description: "Daun sirih (Piper betle) adalah bagian dari tanaman sirih yang tumbuh subur di berbagai daerah tropis...",
        benefit: "1. Menyehatkan saluran pencernaan...\n2. Mengatasi sembelit...\n3. Menjaga kesehatan mulut dan gigi..."
    }
};

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
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
            .div(tf.scalar(255.0));

        log('Image processed into tensor');

        await loadModel();
        const predictions = await model.predict(tensor).data();
        log('Model prediction completed');

        // Get the index of the highest prediction
        const maxPredictionIndex = predictions.indexOf(Math.max(...predictions));
        const predictedPlant = Object.keys(plantInfo)[maxPredictionIndex];
        const { binomial, description, benefit } = plantInfo[predictedPlant];

        log(`Predicted plant: ${predictedPlant}`);

        return h.response({ 
            plant: predictedPlant,
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
