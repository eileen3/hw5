def build_decision_tree(training_set, labels):
	features = range(len(training_set[0]))
	root = Node(features)
	for lbl in labels:
		root.add_pattern(lbl)
	tree = DecisionTree(root)
	node_queue = [root]
	while len(node_queue) > 0:
		curr_node = node_queue.pop(0)
		if len(curr_node.features_left) == 0:
			print 'no features left'
			if curr_node.spam > curr_node.not_spam:
				curr_node.set_classification(1)
			else:
				curr_node.set_classification(0)
			continue
		if curr_node.is_spam():
			curr_node.set_classification(1)
			continue
		elif curr_node.is_not_spam():
			curr_node.set_classification(0)
			continue
		left_child, right_child = propogate_tree(
			curr_node, training_set, labels)
		if right_child is None or left_child is None:
			if curr_node.spam > curr_node.not_spam:
				curr_node.set_classification(1)
			else:
				curr_node.set_classification(0)
			continue
		else:
			node_queue += [left_child, right_child]
		tree.add_node(left_child)
		tree.add_node(right_child)
	return tree


def propogate_tree(curr_node, training_set, labels):
	value_index = 0
	highest_entropy = -float('inf')
	best_left = None
	best_right = None
	for feature in curr_node.features_left:
		value = training_set[value_index][feature]
		f = list(curr_node.features_left)
		f.remove(feature)
		node_left = Node(f)
		node_right = Node(f)
		for num_sample in range(len(training_set)):
			if training_set[num_sample][feature] < value:
				node_left.add_pattern(labels[num_sample])
			else:
				node_right.add_pattern(labels[num_sample])
		if node_left.is_empty() or node_right.is_empty():
			continue
		if node_left.is_spam():
			node_left.set_classification(1)
		elif node_left.is_not_spam():
			node_left.set_classification(0)
		if node_right.is_spam():
			node_right.set_classification(1)
		elif node_right.is_not_spam():
			node_right.set_classification(0)
		curr_node.set_children(node_left, node_right)
		curr_entropy = calc_entropy_change(curr_node)
		if curr_entropy > highest_entropy:
			curr_node.set_feature(feature)
			curr_node.set_split_value(value)
			best_left = node_left
			best_right = node_right
		value_index += 1
	curr_node.set_children(best_left, best_right)
	return (curr_node.left_child, curr_node.right_child)


def calc_entropy_change(node):
	p_l = node.left_child.get_total_patt() / node.get_total_patt()
	entropy_node = None
	left_entropy = None
	right_entropy = None
	if node.classification is not None:
		entropy_node = 0
	else:
		entropy_node = calc_entropy(node)
	if node.left_child.classification is not None:
		left_entropy = 0
	else:
		left_entropy = calc_entropy(node.left_child)
	if node.right_child.classification is not None:
		right_entropy = 0
	else:
		right_entropy = calc_entropy(node.left_child)
	return entropy_node - p_l * left_entropy - \
		(1 - p_l) * right_entropy


def calc_entropy(node):
	return -(node.get_frac_spam() * math.log(node.get_frac_spam(), 2) + \
		node.get_frac_not_spam() * math.log(node.get_frac_not_spam(), 2))


class DecisionTree:
	def __init__(self, root):
		self.root = root
		self.nodes = []

	def add_node(self, node):
		self.nodes += [node]

	def get_leaf_nodes(self):
		leaves = []
		for n in self.nodes:
			if n.is_leaf():
				leaves += [n]
		return leaves

	def display(self):
		print str(self.get_leaf_nodes()) + '\n\n'


class Node:
	def __init__(self, features):
		self.feature = None
		self.split_value = None
		self.classification = None
		self.spam = 0
		self.not_spam = 0
		self.total = 0
		self.left_child = None
		self.right_child = None
		self.features_left = features

	def is_leaf(self):
		return self.classification is not None

	def add_pattern(self, label):
		if label == 1:
			self.spam += 1
		else:
			self.not_spam += 1
		self.total += 1

	def set_children(self, left, right):
		self.left_child = left
		self.right_child = right

	def set_classification(self, num):
		self.classification = num

	def set_feature(self, feature):
		self.feature = feature
		self.features_left.remove(feature)

	def set_split_value(self, split_value):
		self.split_value = split_value

	def is_empty(self):
		return self.total == 0

	def is_spam(self):
		return self.spam != 0 and self.not_spam == 0

	def is_not_spam(self):
		return self.spam == 0 and self.not_spam != 0

	def get_frac_spam(self):
		return float(self.spam) / (self.total)

	def get_frac_not_spam(self):
		return float(self.not_spam) / (self.total)

	def get_total_patt(self):
		return self.spam + self.not_spam

	def __repr__(self):
		return ''.join(['feature=', str(self.feature), ';split_value=', str(self.split_value),
			';classification=', str(self.classification),
			';spam=', str(self.spam), ';not_spam=', str(self.not_spam)])
		