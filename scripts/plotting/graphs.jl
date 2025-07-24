####################################################
### FILE COPIED FROM MathisClautrier/NSGA2CGP.jl ###
####################################################

using CartesianGeneticProgramming
# using LightGraphs
using Graphs
using MetaGraphs
using TikzGraphs
using TikzPictures
using LaTeXStrings


function to_graph(ind::CGPInd; active_outputs=trues(ind.n_out), mask_inputs=true)
    actives = [n.active for n in ind.nodes]
    actives[1:ind.n_in] .= true
    vids = findall(actives)
    #pos = get_positions(c)
    mg = MetaDiGraph(SimpleDiGraph())
    add_vertices!(mg, length(vids)+sum(active_outputs))#c.nout)

    # inputs = [ # omega, P, phi, 2phi, Dir, Hsb, Tp, E, Sla, rivdis
    #     L"$\Omega$",#1
    #     L"$P^{0.5}$",#2
    #     L"$\phi$",#3
    #     L"$2\phi$",#4
    #     L"Dir",#5
    #     L"H_{s,b}",#6
    #     L"T_p",#7
    #     L"Sla",#9
    #     L"rivdis",#10
    #     # L"X"
    # ]

    # inputs = [ #_omega, _p, _phi, 2*_phi, _dir, _hs, _tp, _sla, _rivdis
    #     L"$\Omega$",#1
    #     L"$P$",#2
    #     L"$\phi$",#3
    #     L"$2\phi$",#4
    #     L"Dir",#5
    #     L"H_{s,b}",#6
    #     L"T_p",#7
    #     L"SL",#9
    #     L"FW",#10
    # ]

    inputs = [
        L"P",#1
        L"SL",#9
        L"FW",#9
    ]
    
    outputs = [
    L"$dx/dt$",#1
    ]

    if mask_inputs
        temp_labels = similar(inputs)
        temp_labels .= ""
        active_indices = get_output_trace(ind)[get_output_trace(ind) .<= length(inputs)]
        temp_labels[active_indices] .= inputs[active_indices] 
        inputs = temp_labels
    end

    ### omega eq
    # inputs = [
    # L"\Omega",#1
    # L"\phi",#3
    # L"2\phi"#4
    # ]
    # outputs = [
    # L"\Omega_{eq}"#1
    # ]

    # ### F
    # inputs = [
    #     L"\Omega",#1
    #     L"\Omega_{eq}",#1
    #     L"$P^{0.5}$",#2
    # ]
    # outputs = [
    #     # L"{\Delta \Omega} \over {\sigma \Delta \Omega}"
    #     L"F"
    # ]

    ### r
    # inputs = [
    #     # L"\Omega",#1
    #     # L"\Omega_{eq}",#1
    #     L"$F^-$",#2
    #     L"$F^+$",#2
    #     L"$r$",#2
    # ]
    # outputs = [
    #     # L"{\Delta \Omega} \over {\sigma \Delta \Omega}"
    #     L"$dx \over dt$",
    #     # L"F^-",
    #     # L"F^+",
    # ]
    
    for i in 1:ind.n_in
        # println(LaTeXString(string("\$in_{", i, "}\$")), inputs[i])
        # set_prop!(mg, i, :name, LaTeXString(string("\$in_{", i, "}\$")))
        set_prop!(mg, i, :name, inputs[i])
        set_prop!(mg, i, :type, 0)
    end

    labels_dict = Dict()
    edges_dict = Dict()
    for vi in (ind.n_in+1):length(vids)
        v = vids[vi]
        n = ind.nodes[v]

        # if split(repr(n.f), ".")[end] != "f_moving_sum"
        #     f_name = split(split(repr(n.f), ".")[end], "_")[end]
        # else
        #     f_name = "movingsum"
        # end

        f_name = split(repr(n.f), ".")[end]
        f_name = replace(replace(f_name, "f_"=>""), "_"=>"\\_")
        # f_name = replace(f_name, "_"=>"\\_")
        # f_name = string(v, ": ", f_name)

        #if f_name == "const"
        #    set_prop!(mg, vi, :name, LaTeXString(@sprintf("%0.2f", n.p)))
        #else

        # if f_name == "irange"
        #     f_name = "indices"
        # elseif f_name == "div"
        #     f_name = L"x \over y"
        # elseif f_name == "tpow"
        #     f_name = L"10^x"
        # elseif f_name == "negate"
        #     f_name = L"-x"
        # end

        println(n.f, "  -  ", f_name)
        set_prop!(mg, vi, :name, f_name) # LaTeXString(f_name))
        #end
        set_prop!(mg, vi, :function, n.f)
        set_prop!(mg, vi, :type, 2)
        #set_prop!(mg, vi, :param, n.p)
        cx = findfirst(x-> x==n.x,vids)
        cy = findfirst(x-> x==n.y,vids)
        if cx == cy
            if cx == vi
                labels_dict[(cx,vi)] = "recurrent"
                edges_dict[(cx,vi)] = "loop right"
            else
                labels_dict[(cx,vi)] = "x"
                edges_dict[(cx,vi)] = "black"
            end
            add_edge!(mg, Edge(cx, vi))
            set_prop!(mg, cx, vi, :ci, 3)
        else
            if cx == vi
                labels_dict[(cx,vi)] = "recurrent"
                edges_dict[(cx,vi)] = "loop right"
            else
                labels_dict[(cx,vi)] = "x"
                edges_dict[(cx,vi)] = "black"
            end
            add_edge!(mg, Edge(cx, vi))
            set_prop!(mg, cx, vi, :ci, 1)
            if typeof(cy)!=Nothing && CGPFunctions.arity[split(repr(n.f), ".")[end]] > 1
                # add to dict (cy,vi) 
                labels_dict[(cy,vi)] = "y"
                edges_dict[(cy,vi)] = "blue"
                add_edge!(mg, Edge(cy, vi))
                set_prop!(mg, cy, vi, :ci, 2)
            end
        end
    end
    nid_count = 1
    for o in 1:ind.n_out
        if active_outputs[o]
            nid = length(vids)+nid_count
            # set_prop!(mg, nid, :name, LaTeXString(string("\$out_{", o, "}\$")))
            set_prop!(mg, nid, :name, outputs[o])
            set_prop!(mg, nid, :type, 1)
            oid = findfirst(x->x== ind.outputs[o],vids)
            add_edge!(mg, Edge(oid, nid))
            set_prop!(mg, nid, oid, :ci, 0)
            nid_count += 1
        end
    end
    mg, labels_dict, edges_dict
end

function chromo_draw(ind::CGPInd, file::String="graph"; active_outputs=trues(ind.n_out), extension="pdf")
    mg, labels_dict, edges_dict = to_graph(ind, active_outputs=active_outputs)
    names = map(x->get_prop(mg, x, :name), 1:nv(mg))
    t = TikzGraphs.plot(mg.graph, names, edge_styles=edges_dict)
    # t = TikzGraphs.plot(mg.graph, names, edge_labels=labels_dict, edge_styles=edges_dict)
    if extension == "pdf"
        TikzPictures.save(TikzPictures.PDF(file), t)
    elseif extension == "svg"
        TikzPictures.save(TikzPictures.SVG(file), t)
    else
        println("chromo_draw: Extension unknown")
    end
end
