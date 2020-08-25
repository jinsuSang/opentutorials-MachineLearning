const tf = require('@tensorflow/tfjs')
/**
   * features와 label을 tensor2d로 반환합니다.
   * @param {tf.data.Dataset} DataSet 
   * @param {Integer} batchNumber 
   * @param {Integer} start  
   * @param {Integer} end 
   */
const createTestDataSet = async (DataSet, batchNumber, start, end) => {
  if (start > end) {
    throw new Error("Error: 'start' is bigger than 'end'")
  }
  if (start < 0 || end < 0) {
    throw new Error("Error: 'start' and 'end' start from zero")
  }

  const dataArr = await DataSet.toArrayForTest()

  if (dataArr.length < batchNumber) {
    throw new Error("Error: Exceed batch number")
  }

  const batchSize = dataArr[batchNumber].xs.shape[0]
  if (end > batchSize) {
    throw new Error('Error: Exceed batch size')
  }

  let range = []
  for (let i = start; i < end; i++) {
    range.push(i)
  }
  const tensor1dRange = tf.tensor1d(range, 'int32')

  const xs = tf.tidy(() => {
    const tensor2dXs = dataArr[batchNumber].xs
    const XsTestDataset = tf.gather(tensor2dXs, tensor1dRange)
    return XsTestDataset
  })

  const ys = tf.tidy(() => {
    const tensor2dYs = dataArr[batchNumber].ys
    const YsTestDataset = tf.gather(tensor2dYs, tensor1dRange)
    return YsTestDataset
  })
  return { xs, ys }
}

module.exports = createTestDataSet