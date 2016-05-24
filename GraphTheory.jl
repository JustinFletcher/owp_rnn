
module GraphTheory

	export Graph, Vertex, DirectedEdge, V, E, w, n, K, ω

    type Vertex

        id::Int

        function Vertex(id::Int)
            new(id)
        end

    end

    type DirectedEdge

        head::Vertex
        tail::Vertex
        w::Float64

        function DirectedEdge(head::Vertex, tail::Vertex, w::Float64)
            new(head, tail, w)
        end

    end

    type Walk

        function Walk()
            new()
        end

    end

    type Graph

        V
        E

        function Graph(ω)

            V = Vertex[Vertex(v) for v in 1:length(ω[1,:])]
            E = DirectedEdge[]
            for u in V
                for v in V
                    push!(E, DirectedEdge(u,v,ω[u.id,v.id]))
                end
            end
            new(V,E)
        end
    end


    function K(n, randfun=rand)
        return(Graph(randfun((n,n)).*abs(eye(n)-1)))
    end

    function V(G::Graph)
        return(G.V)
    end

    function E(G::Graph)
        return(G.E)
    end

    function w(e::DirectedEdge)
        return(e.w)
    end

    function n(G::Graph)
        return(length(V(G)))
    end

    function w(G::Graph)

        ω = zeros((n(G), n(G)))
        for e in E(G)
            ω[e.tail.id,e.head.id] = w(e)
        end

        return(ω)
    end
end






