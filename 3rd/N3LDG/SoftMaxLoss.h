#ifndef _SOFTMAXLOSS_H_
#define _SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"


class SoftMaxLoss {
public:
	inline dtype loss(PNode x, const string &answer, Metric& eval, int batchsize = 1) {
		int nDim = x->dim;
		int labelsize = 3;
		if (labelsize != nDim) {
			std::cerr << "softmax_loss error: dim size invalid" << std::endl;
			return -1.0;
		}

		NRVec<dtype> scores(nDim);
		int ans;
		if (answer == "AGAINST") ans = 0;
		if (answer == "FAVOR") ans = 1;
		if (answer == "NONE") ans = 2;
		dtype cost = 0.0;
		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			//if (bTargetInTweet && i == 2) continue;
			if (optLabel < 0 || x->val[i] > x->val[optLabel])
				optLabel = i;
		}

		dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
		for (int i = 0; i < nDim; ++i) {
			scores[i] = exp(x->val[i] - maxScore);
			if (ans == i)
				sum1 += scores[i];
			sum2 += scores[i];

		}
		cost += (log(sum2) - log(sum1)) / batchsize;
		if (ans == optLabel)
			eval.correct_label_count++;
		eval.overall_label_count++;

		for (int i = 0; i < nDim; ++i) {
			float t = ans == i ? 1.0 : 0.0;
			x->loss[i] = (scores[i] / sum2 - t) / batchsize;
		}


		return cost;

	}

	inline dtype predict(PNode x, int& y, bool bTargetInTweet) {
		int nDim = x->dim;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (bTargetInTweet && i == 2) continue;

			if (optLabel < 0 || x->val[i] >  x->val[optLabel])
				optLabel = i;
		}

		dtype prob = 0.0;
		dtype sum = 0.0;
		NRVec<dtype> scores(nDim);
		dtype maxScore = x->val[optLabel];
		for (int i = 0; i < nDim; ++i) {
			scores[i] = exp(x->val[i] - maxScore);
			sum += scores[i];
		}
		prob = scores[optLabel] / sum;
		y = optLabel;
		return prob;
	}

	inline dtype cost(PNode x, const string &answer, int batchsize = 1) {
		int nDim = x->dim;
		int labelsize = 3;
		if (labelsize != nDim) {
			std::cerr << "softmax_loss error: dim size invalid" << std::endl;
			return -1.0;
		}
		int ans;
		if (answer == "AGAINST") ans = 0;
		if (answer == "FAVOR") ans = 1;
		if (answer == "NONE") ans = 2;
		NRVec<dtype> scores(nDim);

		dtype cost = 0.0;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (optLabel < 0 || x->val[i] > x->val[optLabel])
				optLabel = i;
		}

		dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
		for (int i = 0; i < nDim; ++i) {
			scores[i] = -1e10;
			if (answer[i] >= 0) {
				scores[i] = exp(x->val[i] - maxScore);
				if (ans == i)
					sum1 += scores[i];
				sum2 += scores[i];
			}
		}
		cost += (log(sum2) - log(sum1)) / batchsize;
		return cost;
	}

};


#endif /* _SOFTMAXLOSS_H_ */
