from models import stream, clinic, patient

def setup(env, ward, t_inter, pretime, optime, posttime, endtime):
    while env.now < endtime:
        interarrivaltime = t_inter.new()
        pre = pretime.new()
        op = optime.new()
        post = posttime.new()
        new = patient(env, ward, pre, op, post)
        env.process(new.run())
        yield env.timeout(interarrivaltime)