'use strict';

const Hapi = require('@hapi/hapi');
const Inert = require('@hapi/inert');
const routes = require('./routes/classify');
require('dotenv').config();
const log = console.log;

const init = async () => {
    const server = Hapi.server({
        port: 3000,
        host: 'localhost'
    });

    await server.register(Inert);

    server.route(routes);

    await server.start();
    log('Server running on %s', server.info.uri);
};

process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

init();
