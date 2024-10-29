'use strict'

const tracer = require('dd-trace').init()
tracer.use('express', false)
tracer.use('http', false)
tracer.use('dns', false)

const express = require('express');

const app = express();
app.use(express.json());

app.post('/openai/chat_completion', (req, res) => {
});
