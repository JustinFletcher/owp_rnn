function removeDataMean(dataIn, inputCols)
  data = copy(dataIn)
  for col in inputCols
    data[:,col] = (data[:,col].-mean(data[:,col]))
  end
  return(data)
end
