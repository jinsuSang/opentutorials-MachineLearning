const fs = require('fs')
const csv = require('csv-parser')

/**
 * 
 * @param { string } csvFilePath csv 파일 주소
 * 리턴값은 [{'a':'1', 'b':'2'}] 객체 배열 형태입니다.
 */
const readCSV = (csvFilePath) => {
  return new Promise((resolve, reject) => {
    const dataSet = []
    const readStream = fs.createReadStream(csvFilePath)
    readStream.pipe(csv()).on('error', () => { return reject(new Error('Error reading file')) }).on('data', (data) => { dataSet.push(data) }).on('end', () => { resolve(dataSet) })
  })
}

module.exports = readCSV
