
include("$(pwd())//getUniqueElements.jl")

function orthogonalizeDataClasses(data,classCol)

    colVector  = (1:size(data)[2])[(1:size(data)[2]).!=classCol]

    transformedData = copy(data)
    transformedData = transformedData[:, colVector]
    # Get each unique class value
    for class in getUniqueElements(data[:, classCol])
        transformedData = [transformedData (2*int(data[:, classCol] .== class))-1]
    end

    return(float64(transformedData))
end
