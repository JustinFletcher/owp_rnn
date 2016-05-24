function getUniqueElements(vector)

    # Initialize an empty array to stroe the known classes.
    knownList = Array(Any, 0)

    # Iterate over each element in the provided vector.
    for element in vector

        # If the element isn't already in the list...
        if (! in(element, knownList))

            # Append it to the list.
            knownList=vcat(knownList, element)

        end
    end

    return(knownList)

end
