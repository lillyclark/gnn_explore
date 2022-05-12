import Networks
import Policies
import Environments

training = Policies.DeepQ(Environments.SimpleEnvironment)
policy_net = Networks.GCN()
target_net = Networks.GCN()
training.running(policy_net, target_net)
