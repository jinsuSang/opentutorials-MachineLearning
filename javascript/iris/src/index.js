const express = require('express')
const tf = require('@tensorflow/tfjs')
const path = require('path')
const createTestDataSet = require('./utils/createTestDataSet')

const port = process.env.PORT
const publicDirectoryPath = path.join(__dirname + '../public')

const app = express()

app.use(express.static(publicDirectoryPath))

const run = async () => {
  const irisCSVPath = 'file://' + path.join(__dirname, '../data/iris.csv')
  const irisCSV = tf.data.csv(irisCSVPath, { hasHeader: true, columnConfigs: { '품종': { isLabel: true } } })
  const irisDataset = irisCSV.map(({ xs, ys }) => {
    let label = null
    switch (ys['품종']) {
      case 'setosa':
        label = [1, 0, 0]
        break;
      case 'versicolor':
        label = [0, 1, 0]
        break;
      case 'virginica':
        label = [0, 0, 1]
        break;
      default:
        break;
    }
    return { xs: Object.values(xs), ys: label }
  }).batch(16).shuffle(10)

  const input = tf.input({ shape: [4] })

  let hidden = tf.layers.dense({ units: 8 }).apply(input)
  hidden = tf.layers.layerNormalization().apply(hidden)
  hidden = tf.layers.activation({ activation: 'selu' }).apply(hidden)

  hidden = tf.layers.dense({ units: 8 }).apply(hidden)
  hidden = tf.layers.layerNormalization().apply(hidden)
  hidden = tf.layers.activation({ activation: 'selu' }).apply(hidden)

  hidden = tf.layers.dense({ units: 8 }).apply(hidden)
  hidden = tf.layers.layerNormalization().apply(hidden)
  hidden = tf.layers.activation({ activation: 'selu' }).apply(hidden)

  const output = tf.layers.activation({ activation: 'softmax' }).apply(tf.layers.dense({ units: 3 }).apply(hidden))

  const model = tf.model({ inputs: input, outputs: output })
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: 'accuracy' })
  model.summary()


  await model.fitDataset(irisDataset, {
    epochs: 1000
  })

  await model.fitDataset(irisDataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`epoch${epoch}: loss= ${logs.loss}, acc= ${logs.acc}`);
      }
    }
  })

  const testDataset = await createTestDataSet(irisDataset, 0, 0, 8)
  model.predict(testDataset.xs).print()
  testDataset.ys.print()
  model.getWeights()[0].print()
}

run()

app.listen(port, () => {
  console.log('Iris Classification Server is up to start')
})