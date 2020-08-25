'use strict'
const express = require('express')
const path = require('path')
const tf = require('@tensorflow/tfjs')
const createTestDataSet = require('./utils/createTestDataSet')

const port = process.env.PORT
const publicDirectoryPath = path.join(__dirname, '../public')

const app = express()
app.use(express.static(publicDirectoryPath))

const run = async () => {
  const bostonPriceCSV = tf.data.csv(
    'file://' + path.join(__dirname, '../data/boston.csv'),
    { columnConfigs: { medv: { isLabel: true } } }
  )

  const bostonPriceDataset = bostonPriceCSV.map(({ xs, ys }) => {
    return { xs: Object.values(xs), ys: Object.values(ys) }
  }).batch(50).shuffle(10)

  const input = tf.input({ shape: [13] })

  let hidden = tf.layers.dense({ units: 8 }).apply(input)
  hidden = tf.layers.batchNormalization().apply(hidden)
  hidden = tf.layers.activation({ activation: 'selu' }).apply(hidden)

  hidden = tf.layers.dense({ units: 4 }).apply(hidden)
  hidden = tf.layers.batchNormalization().apply(hidden)
  hidden = tf.layers.activation({ activation: 'selu' }).apply(hidden)

  hidden = tf.layers.dense({ units: 2 }).apply(hidden)
  hidden = tf.layers.batchNormalization().apply(hidden)
  hidden = tf.layers.activation({ activation: 'selu' }).apply(hidden)

  const output = tf.layers.dense({ units: 1 }).apply(hidden)

  const model = tf.model({
    inputs: input,
    outputs: output
  })
  model.compile({ optimizer: tf.train.adam(0.01), loss: tf.losses.meanSquaredError })
  model.summary()

  await model.fitDataset(bostonPriceDataset, {
    epochs: 3000
  })


  await model.fitDataset(bostonPriceDataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch + ':' + logs.loss);
      }
    }
  })

  try {
    const testDataset = await createTestDataSet(bostonPriceDataset, 0, 0, 5)
    model.predict(testDataset.xs).print()
    testDataset.ys.print()
    model.getWeights()[0].print()
  } catch (error) {
    console.error(error.message)
  }

}

run()

app.listen(port, () => {
  console.log('Boston house price prediction Server is up to start')
})