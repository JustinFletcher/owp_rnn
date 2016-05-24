function normalizeData(dataIn)
  data = copy(dataIn)
  for col in 1:size(data)[2]
    data[:,col] = ((data[:,col]./maximum(data[:,col])))
  end
  return(data)
end
