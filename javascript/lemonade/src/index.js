'use strict'
const express = require('express')
const path = require('path')
const tf = require('@tensorflow/tfjs')
const readCSV = require('./utils/readCSV')


const port = process.env.PORT
const publicDirectoryPath = path.join(__dirname, '../public')

const app = express()
app.use(express.static(publicDirectoryPath))

// tensorflow
const run = async () => {
  const lemonadeCSV = await readCSV('./data/lemonade.csv')

  // 데이터를 tensor 형태로 만들기
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

  // 모델 만들기
  const input = tf.input({ shape: [1] })
  const output = tf.layers.dense({ units: 1 }).apply(input)
  const model = tf.model({
    inputs: input,
    outputs: output
  })

  model.compile({ optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError })
  model.summary()

  // 모델 학습
  const { history: { loss: loss } } = await model.fit(independent, dependent, { epochs: 15000 })

  // 마지막 loss 출력
  console.log(loss[loss.length - 1])

  // 판매량 예측
  model.predict(tf.tensor1d([40])).print()
}

run()

app.listen(port, () => {
  console.log('ReadCSV tensorflow Server is up to start')
})  