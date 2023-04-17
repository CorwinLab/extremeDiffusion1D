import numpy as np
from numba import njit, jit
import csv
import os
import sys


@jit
def iteratePDF(right, left, quantile, dist="beta", params=1):
	if dist == "beta":
		if params == 1:
			# This is to only generate random numbers on the odd values
			# which should be populated
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
		elif params == 0:
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
			biases = np.array([np.round(i) for i in biases])
		elif params == np.inf:
			biases = np.ones(right.shape) / 2
		else:
			biases = np.zeros(right.size)
			rand_vals = np.random.beta(params, params, size=right[::2].size)
			biases[::2] = rand_vals

	elif dist == "delta":
		biases = np.zeros(right.size)
		rand_nums = np.random.choice(
			np.array([0, 1 / 2, 1]),
			size=right[::2].size,
			p=np.array([params, 1 - 2 * params, params]),
		)
		biases[::2] = rand_nums

	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	quantileSet = False

	for i in range(1, right.size - 1):
		# Scattering Model for diffusion
		right_new[i] = right[i - 1] * biases[i - 1] + left[i - 1] * (1 - biases[i - 1])
		left_new[i] = left[i + 1] * biases[i + 1] + right[i + 1] * (1 - biases[i + 1])

		# RWRE regular diffusion
		# right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (biases[i-1])
		# left_new[i] = left[i+1] * (1-biases[i+1]) + right[i+1] * (1-biases[i+1])

		cdf_new += right_new[i] + left_new[i]
		if (1 - cdf_new <= quantile) and not quantileSet:
			pos = i - (right_new.size // 2)
			quantileSet = True

	return right_new, left_new, pos


@jit
def iteratePDFGetVelocities(right, left, xval, dist="beta", params=1):
	''' Note: xvals should be in decending order '''
	if dist == "beta":
		if params == 1:
			# This is to only generate random numbers on the odd values
			# which should be populated
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
		elif params == 0:
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
			biases = np.array([np.round(i) for i in biases])
		elif params == np.inf:
			biases = np.ones(right.shape) / 2
		else:
			biases = np.zeros(right.size)
			rand_vals = np.random.beta(params, params, size=right[::2].size)
			biases[::2] = rand_vals

	elif dist == "delta":
		biases = np.zeros(right.size)
		rand_nums = np.random.choice(
			np.array([0, 1 / 2, 1]),
			size=right[::2].size,
			p=np.array([params, 1 - 2 * params, params]),
		)
		biases[::2] = rand_nums

	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	delta_new = 0 

	prob = np.nan
	delta = np.nan

	for i in range(1, right.size - 1):
		# Scattering Model for diffusion
		right_new[i] = right[i - 1] * biases[i - 1] + left[i - 1] * (1 - biases[i - 1])
		left_new[i] = left[i + 1] * biases[i + 1] + right[i + 1] * (1 - biases[i + 1])

		# RWRE regular diffusion
		# right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (biases[i-1])
		# left_new[i] = left[i+1] * (1-biases[i+1]) + right[i+1] * (1-biases[i+1])

		cdf_new += right_new[i] + left_new[i]
		delta_new += right_new[i] - left_new[i]
		
		pos = i - (right_new.size // 2)
		if pos == xval:
			prob = cdf_new[i] 
			delta = delta_new[i]

	return right_new, left_new, prob, delta


@jit
def iteratePDFModified(right, left, quantile, dist="beta", params=1):
	if dist == "beta":
		if params == 1:
			# This is to only generate random numbers on the odd values
			# which should be populated
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
		elif params == 0:
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
			biases = np.array([np.round(i) for i in biases])
		elif params == np.inf:
			biases = np.ones(right.shape) / 2
		else:
			biases = np.zeros(right.size)
			rand_vals = np.random.beta(params, params, size=right[::2].size)
			biases[::2] = rand_vals

	elif dist == "delta":
		biases = np.zeros(right.size)
		rand_nums = np.random.choice(
			np.array([0, 1 / 2, 1]),
			size=right[::2].size,
			p=np.array([params, 1 - 2 * params, params]),
		)
		biases[::2] = rand_nums

	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	quantileSet = False

	for i in range(1, right.size - 1):
		# Modified Scattering Model for diffusion
		right_new[i] = right[i - 1] * biases[i - 1] + right[i + 1] * (1 - biases[i + 1]) + left[i-1] * (1-biases[i-1])
		left_new[i] = left[i + 1] * biases[i + 1]

		# RWRE regular diffusion
		# right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (biases[i-1])
		# left_new[i] = left[i+1] * (1-biases[i+1]) + right[i+1] * (1-biases[i+1])

		cdf_new += right_new[i] + left_new[i]
		if (1 - cdf_new <= quantile) and not quantileSet:
			pos = i - (right_new.size // 2)
			quantileSet = True
		
	return right_new, left_new, pos

@njit
def generalizedPDF(prr, pll, prl, plr, quantile):
	biases = np.random.uniform(0, 1, size=prr.shape)

	prr_new = np.zeros(prr.shape)
	pll_new = np.zeros(pll.shape)
	prl_new = np.zeros(prl.shape)
	plr_new = np.zeros(plr.shape)

	cdf_new = 0
	quantileSet = False

	for i in range(1, prr.size - 1):
		pll_new[i] = pll[i + 1] * biases[i + 1] + prl[i + 1] * biases[i + 1]
		prr_new[i] = prr[i - 1] * biases[i - 1] + plr[i - 1] * biases[i - 1]
		prl_new[i] = pll[i - 1] * (1 - biases[i - 1]) + prl[i - 1] * (1 - biases[i - 1]) # I'm not sure that this is correct, I think it should be plr
		plr_new[i] = prr[i + 1] * (1 - biases[i + 1]) + plr[i + 1] * (1 - biases[i + 1]) # And I think this should be prl

		cdf_new += prr_new[i] + pll_new[i] + prl_new[i] + plr_new[i]
		if (1 - cdf_new <= quantile) and not quantileSet:
			pos = i - (prr_new.size // 2)
			quantileSet = True
			
	return prr_new, pll_new, prl_new, plr_new, pos


def cyclicPDF(p1, p2, p3, p4, quantile):
	p1_new = np.zeros(p1.shape)
	p2_new = np.zeros(p2.shape)
	p3_new = np.zeros(p3.shape)
	p4_new = np.zeros(p4.shape)

	biases = np.random.uniform(0, 1, p1_new.size) 
	cdf_new = 0
	quantileSet = False
	for i in range(1, len(p1_new)-1):
		p1_new[i] = p1[i-1] * biases[i-1] + p2[i-1] * (1-biases[i-1])
		p2_new[i] = p2[i+1] * biases[i+1] + p3[i+1] * (1-biases[i+1])
		p3_new[i] = p3[i-1] * biases[i-1] + p4[i-1] * (1-biases[i-1])
		p4_new[i] = p4[i+1] * biases[i+1] + p1[i+1] * (1-biases[i+1])
		
		cdf_new += p1_new[i] + p2_new[i] + p3_new[i] + p4_new[i]
		if (1 - cdf_new <= quantile) and not quantileSet:
			pos = i - (p1_new.size // 2)
			quantileSet = True

	return p1_new, p2_new, p3_new, p4_new, pos

@jit
def cyclicDirichletPDF(p1, p2, p3, p4, quantile):
	p1_new = np.zeros(p1.shape)
	p2_new = np.zeros(p2.shape)
	p3_new = np.zeros(p3.shape)
	p4_new = np.zeros(p4.shape)

	biases = np.random.dirichlet([1, 1, 1, 1], p1_new.size)
	cdf_new = 0
	quantileSet = False
	for i in range(1, len(p1_new)-1):
		p1_new[i] = p1[i-1] * biases[i-1][0] + p3[i-1] * biases[i-1][1] + p2[i-1] * biases[i-1][2] + p4[i-1] * biases[i-1][3]
		p2_new[i] = p2[i+1] * biases[i+1][0] + p1[i+1] * biases[i+1][1] + p4[i+1] * biases[i+1][2] + p3[i+1] * biases[i+1][3]
		p3_new[i] = p3[i-1] * biases[i-1][0] + p4[i-1] * biases[i-1][1] + p1[i-1] * biases[i-1][2] + p2[i-1] * biases[i-1][3]
		p4_new[i] = p4[i+1] * biases[i+1][0] + p2[i+1] * biases[i+1][1] + p3[i+1] * biases[i+1][2] + p1[i+1] * biases[i+1][3]
		
		cdf_new += p1_new[i] + p2_new[i] + p3_new[i] + p4_new[i]
		if (1 - cdf_new <= quantile) and not quantileSet:
			pos = i - (p1_new.size // 2)
			quantileSet = True

	return p1_new, p2_new, p3_new, p4_new, pos


def evolveAndGetQuantile(times, N, size, dist, params, save_file):
	right = np.zeros(size + 1)
	left = np.zeros(size + 1)

	# Start with all the particles moving to the right
	right[right.size // 2] = 1

	write_header = True
	# Check if save file has already been created and make sure we don't
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=",")
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False

	f = open(save_file, "a")
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(["Time", "Position"])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		right_new, left_new, pos = iteratePDF(
			right[size // 2 - t - 2 : size // 2 + t + 3],
			left[size // 2 - t - 2 : size // 2 + t + 3],
			1 / N,
			dist=dist,
			params=params,
		)
		right[size // 2 - t - 2 : size // 2 + t + 3] = right_new
		left[size // 2 - t - 2 : size // 2 + t + 3] = left_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(right + left) - 1) < 1e-10, np.abs(
			np.sum(right + left) - 1
		)

		if t in times:
			writer.writerow([t + 1, pos])
			f.flush()
	f.close()


def evolveAndGetQuantileGeneralized(times, N, size, save_file):
	prr = np.zeros(size + 1)
	pll = np.zeros(size + 1)
	prl = np.zeros(size + 1)
	plr = np.zeros(size + 1)

	# Start with all the particles moving to the right
	prr[prr.size // 2] = 1

	write_header = True
	# Check if save file has already been created and make sure we don't
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=",")
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False

	f = open(save_file, "a")
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(["Time", "Position"])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		prr_new, pll_new, prl_new, plr_new, pos = cyclicDirichletPDF(
			prr[size // 2 - t - 2 : size // 2 + t + 3],
			pll[size // 2 - t - 2 : size // 2 + t + 3],
			prl[size // 2 - t - 2 : size // 2 + t + 3],
			plr[size // 2 - t - 2 : size // 2 + t + 3],
			1 / N,
		)
		prr[size // 2 - t - 2 : size // 2 + t + 3] = prr_new
		pll[size // 2 - t - 2 : size // 2 + t + 3] = pll_new
		prl[size // 2 - t - 2 : size // 2 + t + 3] = prl_new
		plr[size // 2 - t - 2 : size // 2 + t + 3] = plr_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(prr + pll + prl + plr) - 1) < 1e-10, np.abs(
			np.sum(prr + pll + prl + plr) - 1
		)

		if t in times:
			writer.writerow([t + 1, pos])
			f.flush()
	f.close()


def evolveAndGetQuantileCyclic(times, N, size, save_file):
	prr = np.zeros(size + 1)
	pll = np.zeros(size + 1)
	prl = np.zeros(size + 1)
	plr = np.zeros(size + 1)

	# Start with all the particles moving to the right
	prr[prr.size // 2] = 1

	write_header = True
	# Check if save file has already been created and make sure we don't
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=",")
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False

	f = open(save_file, "a")
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(["Time", "Position"])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		prr_new, pll_new, prl_new, plr_new, pos = cyclicPDF(
			prr[size // 2 - t - 2 : size // 2 + t + 3],
			pll[size // 2 - t - 2 : size // 2 + t + 3],
			prl[size // 2 - t - 2 : size // 2 + t + 3],
			plr[size // 2 - t - 2 : size // 2 + t + 3],
			1 / N,
		)
		prr[size // 2 - t - 2 : size // 2 + t + 3] = prr_new
		pll[size // 2 - t - 2 : size // 2 + t + 3] = pll_new
		prl[size // 2 - t - 2 : size // 2 + t + 3] = prl_new
		plr[size // 2 - t - 2 : size // 2 + t + 3] = plr_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(prr + pll + prl + plr) - 1) < 1e-10, np.abs(
			np.sum(prr + pll + prl + plr) - 1
		)

		if t in times:
			writer.writerow([t + 1, pos])
			f.flush()
	f.close()


def evolveAndGetProbs(times, N, size, beta, save_file):
	right = np.zeros(size + 1)
	left = np.zeros(size + 1)

	# Start with all the particles moving to the right
	right[right.size // 2] = 1

	write_header = True
	# Check if save file has already been created and make sure we don't
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=",")
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False

	f = open(save_file, "a")
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(["Time", "Position", "Prob", "Delta"])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		right_new, left_new, pos = iteratePDF(
			right[size // 2 - t - 2 : size // 2 + t + 3],
			left[size // 2 - t - 2 : size // 2 + t + 3],
			1 / N,
			beta=beta,
		)
		right[size // 2 - t - 2 : size // 2 + t + 3] = right_new
		left[size // 2 - t - 2 : size // 2 + t + 3] = left_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(right + left) - 1) < 1e-10, np.abs(
			np.sum(right + left) - 1
		)

		idx = pos + (right.size // 2)
		prob = (right + left)[idx]
		delta = (right - left)[idx]

		if t in times:
			writer.writerow([t + 1, pos, prob, delta])
			f.flush()

	f.close()


def evolveAndGetVelocities(times, vs, size, beta, save_file):
	right = np.zeros(size + 1)
	left = np.zeros(size + 1)

	# Start with all the particles moving to the right
	right[right.size // 2] = 1

	write_header = True
	# Check if save file has already been created and make sure we don't
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=",")
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False

	f = open(save_file, "a")
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(["Time", "Position", "Prob", "Delta"])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		xval = np.floor(vs * t ** (3/4))
		right_new, left_new, prob, delta = iteratePDFGetVelocities(
			right[size // 2 - t - 2 : size // 2 + t + 3],
			left[size // 2 - t - 2 : size // 2 + t + 3],
			xval,
			beta=beta,
		)
		right[size // 2 - t - 2 : size // 2 + t + 3] = right_new
		left[size // 2 - t - 2 : size // 2 + t + 3] = left_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(right + left) - 1) < 1e-10, np.abs(
			np.sum(right + left) - 1
		)

		if t in times:
			writer.writerow([t + 1, xval, prob, delta])
			f.flush()

	f.close()


@jit
def biasingField(xvals, correlation_length):
	grid = np.arange(
		np.min(xvals) - 3 * correlation_length,
		np.max(xvals) + 3 * correlation_length,
		step=1,
	)
	noise = np.random.uniform(0, 1, len(grid))

	kernel_x = np.arange(-3 * correlation_length, 3 * correlation_length, 1)
	kernel = np.exp(-(kernel_x**2) / correlation_length**2)
	field = np.convolve(kernel, noise, "same")

	assert np.all(
		np.diff(grid) > 0
	), "Sampling points on grid are not monotonically increasing"
	field = np.interp(xvals, grid, field)
	scaling_factor = np.sqrt(1 / correlation_length**2 / np.pi)
	field *= scaling_factor
	return field


@jit
def iteratePDFFields(right, left, quantile, xvals, rc=2):
	biases = biasingField(xvals, rc)
	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	quantileSet = False

	for i in range(1, right.size - 1):
		# Scattering Model for diffusion
		right_new[i] = right[i - 1] * biases[i - 1] + left[i - 1] * (1 - biases[i - 1])
		left_new[i] = left[i + 1] * biases[i + 1] + right[i + 1] * (1 - biases[i + 1])

		# RWRE regular diffusion
		# right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (biases[i-1])
		# left_new[i] = left[i+1] * (1-biases[i+1]) + right[i+1] * (1-biases[i+1])

		cdf_new += right_new[i] + left_new[i]
		if (1 - cdf_new <= quantile) and not quantileSet:
			pos = i - (right_new.size // 2)
			quantileSet = True

	return right_new, left_new, pos
