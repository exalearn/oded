function mse = MseOfFilter(ryy_trace, ryx, rxx, lin_filter)
    mse = ryy_trace-2*sum(sum(ryx.*lin_filter))+sum(sum((lin_filter*rxx).*lin_filter));
end