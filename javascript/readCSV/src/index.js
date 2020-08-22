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
  let result = await readCSVtoArr('./data/lemonade.csv')
  console.log(result)
}

run()


app.listen(port, () => {
  console.log('ReadCSV tensorflow Server is up to start')
})  