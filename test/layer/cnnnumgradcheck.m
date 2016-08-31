function cnnnumgradcheck(net, x, y)
    epsilon = 1e-4;
    er      = 1e-8;
    n = numel(net.layers);
    for j = 1 : numel(net.ffb)
        net_m = net; net_p = net;
        net_p.ffb(j) = net_m.ffb(j) + epsilon;
        net_m.ffb(j) = net_m.ffb(j) - epsilon;
        net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
        net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
        d = (net_p.L - net_m.L) / (2 * epsilon);
        e = abs(d - net.dffb(j));
        if e > er
            e
            d / net.dffb(j)
            error('numerical gradient checking failed');
        end
    end

    for i = 1 : size(net.ffW, 1)
        for u = 1 : size(net.ffW, 2)
            net_m = net; net_p = net;
            net_p.ffW(i, u) = net_m.ffW(i, u) + epsilon;
            net_m.ffW(i, u) = net_m.ffW(i, u) - epsilon;
            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
            d = (net_p.L - net_m.L) / (2 * epsilon);
            e = abs(d - net.dffW(i, u));
            if e > er
                e
                d / net.ffW(i, u)
                error('numerical gradient checking failed');
            end
        end
    end

    for idx_layer = n : -1 : 2
        if strcmp(net.layers{idx_layer}.type, 'c')
            for j = 1 : numel(net.layers{idx_layer}.a)
                net_m = net; net_p = net;
                net_p.layers{idx_layer}.b{j} = net_m.layers{idx_layer}.b{j} + epsilon;
                net_m.layers{idx_layer}.b{j} = net_m.layers{idx_layer}.b{j} - epsilon;
                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                d = (net_p.L - net_m.L) / (2 * epsilon);
                e = abs(d - net.layers{idx_layer}.db{j});
                if e > er
                    e
                    d / net.layers{idx_layer}.db{j}
                    error('numerical gradient checking failed');
                end
                for i = 1 : numel(net.layers{idx_layer - 1}.a)
                    for u = 1 : size(net.layers{idx_layer}.k{i}{j}, 1)
                        for v = 1 : size(net.layers{idx_layer}.k{i}{j}, 2)
                            net_m = net; net_p = net;
                            net_p.layers{idx_layer}.k{i}{j}(u, v) = net_p.layers{idx_layer}.k{i}{j}(u, v) + epsilon;
                            net_m.layers{idx_layer}.k{i}{j}(u, v) = net_m.layers{idx_layer}.k{i}{j}(u, v) - epsilon;
                            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                            d = (net_p.L - net_m.L) / (2 * epsilon);
                            e = abs(d - net.layers{idx_layer}.dk{i}{j}(u, v));
                            if e > er
                                error('numerical gradient checking failed');
                            end
                        end
                    end
                end
            end
        elseif strcmp(net.layers{idx_layer}.type, 's')
%            for j = 1 : numel(net.layers{idx_layer}.a)
%                net_m = net; net_p = net;
%                net_p.layers{idx_layer}.b{j} = net_m.layers{idx_layer}.b{j} + epsilon;
%                net_m.layers{idx_layer}.b{j} = net_m.layers{idx_layer}.b{j} - epsilon;
%                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
%                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
%                d = (net_p.L - net_m.L) / (2 * epsilon);
%                e = abs(d - net.layers{idx_layer}.db{j});
%                if e > er
%                    error('numerical gradient checking failed');
%                end
%            end
        end
    end
%    keyboard
end
