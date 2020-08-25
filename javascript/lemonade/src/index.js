'use strict'
const express = require('express')
const path = require('path')
const fs = require('fs')
const csv = require('csv-parser')
const tf = require('@tensorflow/tfjs')


const port = process.env.PORT
const publicDirectoryPath = path.join(__dirname, '../public')

const app = express()
app.use(express.static(publicDirectoryPath))

const readCSVtoArr = (csvFilePath) => {
  return new Promise((resolve, reject) => {
    const dataSet = []
    const readStream = fs.createReadStream(csvFilePath)
    readStream.pipe(csv()).on('error', () => { return reject(new Error('Error reading file')) }).on('data', (data) => { dataSet.push(data) }).on('end', () => { resolve(dataSet) })
  })
}

// tensorflow
const run = async () => {
  const lemonadeCSV = await readCSVtoArr('./data/lemonade.csv')

  const temperature = []
  const sales = []
  for (let i = 0; i < lemonadeCSV.length; i++) {
    temperature.push(Number(lemonadeCSV[i]['온도']))
    sales.push(Number(lemonadeCSV[i]['판매량']))
  }

  const independent = tf.tensor1d(temperature)
  const dependent = tf.tensor1d(sales)

  independent.print()
  dependent.print()

  const input = tf.input({ shape: [1] })
  const output = tf.layers.dense({ units: 1 }).apply(input)
  const model = tf.model({
    inputs: input,
    outputs: output
  })

  model.compile({ optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError })
  model.summary()

  const { history: { loss: loss } } = await model.fit(independent, dependent, { epochs: 15000 })
  console.log(loss[loss.length - 1])
  model.predict(tf.tensor1d([40])).print()
}

run()

app.listen(port, () => {
  console.log('ReadCSV tensorflow Server is up to start')
})  