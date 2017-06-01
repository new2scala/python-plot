
import numpy as np

def target_func(v):
    return v[0]**2 + v[1]**2

def grad_target(v):
    return v*2

def grad_target_1(v):
    return (v*2).tolist

def target_taylor_1(v,v0):
    dv = grad_target(v0)
    return dv[0]*(v[0]-v0[0])+dv[1]*(v[1]-v0[1])

def opt_step(v0,ex_vs):
    ex_pts = ex_vs.shape[0]
    tt1 = np.array([target_taylor_1(ex_vs[i],v0) for i in range(0,ex_pts)])
    minidx = tt1.argmin()
    # tv0 = target_func(v0)
    # tmin = target_func(ex_vs[minidx])
    return minidx
    # if tv0 > tmin:
    #     return minidx #ex_vs[]
    # else:
    #     return -1


# tolerance = 1e-6
# def line_search(v0, v1):
#     vv0 = target_func(v0)
#     vv1 = target_func(v1)
#     curr = abs(vv1-vv0)
#
#     while curr > tolerance:
#         v01 = (v0+v1)/2
#         vv01 = target_func(v01)
#         if (vv1 > vv0)


ex_vs = np.array([[0, 1],
                 [1, 0],
                 [3, 1],
                 [1, 3]])

p0 = np.array([3,1])
import scipy.optimize as sci_opt

def find_next_pt(start_pt, extreme_pts):
    idx = opt_step(start_pt, extreme_pts)
    p1 = extreme_pts[idx]
    d = p1-start_pt
    t = sci_opt.line_search(target_func, grad_target, start_pt, d)
    if t[0] is None:
        return None
    else:
        return start_pt + t[0]*d

# def test_func(x):
#     return (x[0])**2+(x[1])**2
#
# def test_grad(x):
#     return 2*x
#
#t = sci_opt.line_search(test_func,test_grad,np.array([1.8,1.7]),np.array([-1.0,-1.0]))

# idx = opt_step(p0, ex_vs)
# p1 = ex_vs[idx]
# d = p1-p0
# t = sci_opt.line_search(target_func, grad_target, p0, d)
#
# p0 = p0 + t[0]*d
# idx = opt_step(p0, ex_vs)
# p1 = ex_vs[idx]
# d = p1-p0
# t = sci_opt.line_search(target_func, grad_target, p0, d)
# print(t)

def run(start_pt, extreme_pts):
    next_pt = find_next_pt(start_pt, extreme_pts)
    while (next_pt is not None):
        tmp = find_next_pt(next_pt, extreme_pts)
        if (tmp is None):
            return next_pt
        next_pt = tmp


print(run(ex_vs[2], ex_vs))

ex_vs = ex_vs - 1

print(run(ex_vs[2], ex_vs))

ex_vs = ex_vs - 3

print(run(ex_vs[2], ex_vs))