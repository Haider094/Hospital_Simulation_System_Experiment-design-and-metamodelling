import random
import simpy

class stream:
    def __init__(self, dist, mean, halfwidth=None):
        self.dist = dist
        self.mean = mean
        self.halfwidth = halfwidth
        if dist == 'exp':
            self.rng = lambda: random.expovariate(1 / mean)
        elif dist == 'unif':
            if halfwidth is None:
                raise ValueError("Uniform requires halfwidth")
            self.rng = lambda: random.uniform(mean - halfwidth, mean + halfwidth)

    def new(self):
        return self.rng()

class clinic(object):
    def __init__(self, env, num_pre, num_op, num_post):
        self.env = env
        self.Preparation = simpy.Resource(env, num_pre)
        self.Operation = simpy.Resource(env, num_op)
        self.Recovery = simpy.Resource(env, num_post)
        self.prewait = 0
        self.opwait = 0
        self.postwait = 0
        self.isblocking = False
        self.isoperational = False

    def reset(self):
        self.prewait = 0
        self.opwait = 0
        self.postwait = 0

    def report(self):
        print(self.prewait, self.opwait, self.postwait)

class patient(object):
    def __init__(self, env, ward, pre, op, post):
        self.env = env
        self.ward = ward
        self.pre = pre
        self.op = op
        self.post = post

    def run(self):
        Prep = self.ward.Preparation
        Op = self.ward.Operation
        Post = self.ward.Recovery

        arrivaltime = self.env.now
        pre_req = Prep.request()
        yield pre_req
        yield self.env.timeout(self.pre)
        op_req = Op.request()
        yield op_req
        self.ward.isoperational = True
        opwaittime = self.env.now
        self.ward.prewait += self.env.now - arrivaltime
        Prep.release(pre_req)

        yield self.env.timeout(self.op)
        self.ward.isoperational = False
        self.ward.isblocking = True
        post_req = Post.request()
        yield post_req
        self.ward.isblocking = False
        postwaittime = self.env.now
        self.ward.opwait += self.env.now - opwaittime
        Op.release(op_req)

        yield self.env.timeout(self.post)
        Post.release(post_req)
        self.ward.postwait += self.env.now - postwaittime  # Added for completeness