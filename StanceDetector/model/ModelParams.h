#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet targetAlpha;

	//LSTM1Params lstm_left_params;
	LSTM1Params lstm_target_left_params;
	LSTM1Params lstm_target_right_params;
	//LSTM1Params lstm_right_params;
	LSTM1Params lstm_tweet_left_params;
	LSTM1Params lstm_tweet_right_params;
	SoftMaxLoss loss;

	UniParams olayer_linear; // output
	BiParams binaryparam;
public:
	Alphabet labelAlpha; // should be initialized outside



public:
	bool initial(HyperParams& opts){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;

		lstm_target_left_params.initial(opts.hiddenSize, opts.wordDim);
		lstm_target_right_params.initial(opts.hiddenSize, opts.wordDim);

		lstm_tweet_left_params.initial(opts.hiddenSize, opts.wordDim);
		lstm_tweet_right_params.initial(opts.hiddenSize, opts.wordDim);

		opts.labelSize = labelAlpha.size();
		//opts.inputSize = opts.hiddenSize;
		opts.inputSize = opts.hiddenSize * 2;
		//opts.inputSize = opts.hiddenSize * 4;
		//opts.inputSize = opts.hiddenSize * 2;
		//opts.inputSize = opts.hiddenSize;
		//opts.inputSize = opts.wordDim;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false);
		binaryparam.initial(opts.labelSize, opts.inputSize, false);

		/*
		vector<dtype> E_val = DEV->to_vector(words.E.val);
		for (int i = 0; i < E_val.size(); i++)
			E_val[i] = 0.1;
		DEV->set(words.E.val, E_val);

		vector<dtype> o_val = DEV->to_vector(olayer_linear.W.val);
		for (int i = 0; i < o_val.size(); i++)
			o_val[i] = 0.02 * i;
		DEV->set(olayer_linear.W.val, o_val);
		*/
		return true;
	}

	bool TestInitial(HyperParams& opts){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.labelSize = labelAlpha.size();
		opts.inputSize = opts.hiddenSize * 3;
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		lstm_target_left_params.exportAdaParams(ada);

		lstm_target_right_params.exportAdaParams(ada);

		lstm_tweet_left_params.exportAdaParams(ada);

		lstm_tweet_right_params.exportAdaParams(ada);

		olayer_linear.exportAdaParams(ada);

		binaryparam.exportAdaParams(ada);
	}

	/*
	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words E");
		
		//checkgrad.add(&bi_linear.W1, "bi_linear.W1");
		//checkgrad.add(&bi_linear.W2, "bi_linear.W2");
		//checkgrad.add(&bi_linear.b, "bi_linear.b");
	
		checkgrad.add(&lstm_target_left_params.input.W1, "lstm_target_left_params.input.W1");
		checkgrad.add(&lstm_target_left_params.input.W2, "lstm_target_left_params.input.W2");
		checkgrad.add(&lstm_target_left_params.input.b, "lstm_target_left_params.input.b");

		checkgrad.add(&lstm_target_left_params.output.W1, "lstm_target_left_params.output.W1");
		checkgrad.add(&lstm_target_left_params.output.W2, "lstm_target_left_params.output.W2");
		checkgrad.add(&lstm_target_left_params.output.b, "lstm_target_left_params.output.b");

		checkgrad.add(&lstm_target_left_params.cell.W1, "lstm_target_left_params.cell.W1");
		checkgrad.add(&lstm_target_left_params.cell.W2, "lstm_target_left_params.cell.W2");
		checkgrad.add(&lstm_target_left_params.cell.b, "lstm_target_left_params.cell.b");


		checkgrad.add(&lstm_target_left_params.forget.W1, "lstm_target_left_params.forget.W1");
		checkgrad.add(&lstm_target_left_params.forget.W2, "lstm_target_left_params.forget.W2");
		checkgrad.add(&lstm_target_left_params.forget.b, "lstm_target_left_params.forget.b");

		checkgrad.add(&lstm_tweet_left_params.input.W1, "lstm_tweet_left_params.input.W1");
		checkgrad.add(&lstm_tweet_left_params.input.W2, "lstm_tweet_left_params.input.W2");
		checkgrad.add(&lstm_tweet_left_params.input.b, "lstm_tweet_left_params.input.b");

		checkgrad.add(&lstm_tweet_left_params.output.W1, "lstm_tweet_left_params.output.W1");
		checkgrad.add(&lstm_tweet_left_params.output.W2, "lstm_tweet_left_params.output.W2");
		checkgrad.add(&lstm_tweet_left_params.output.b, "lstm_tweet_left_params.output.b");

		checkgrad.add(&lstm_tweet_left_params.cell.W1, "lstm_tweet_left_params.cell.W1");
		checkgrad.add(&lstm_tweet_left_params.cell.W2, "lstm_tweet_left_params.cell.W2");
		checkgrad.add(&lstm_tweet_left_params.cell.b, "lstm_tweet_left_params.cell.b");


		checkgrad.add(&lstm_tweet_left_params.forget.W1, "lstm_tweet_left_params.forget.W1");
		checkgrad.add(&lstm_tweet_left_params.forget.W2, "lstm_tweet_left_params.forget.W2");
		checkgrad.add(&lstm_tweet_left_params.forget.b, "lstm_tweet_left_params.forget.b");

		checkgrad.add(&olayer_linear.W, "output layer W");
	}
	*/
	// will add it later
	void saveModel(std::ofstream &os) const{
		wordAlpha.write(os);
		words.save(os);
		olayer_linear.save(os);
		binaryparam.save(os);
		labelAlpha.write(os);
	}

	void loadModel(std::ifstream &is){
		wordAlpha.read(is);
		words.load(is, &wordAlpha);
		olayer_linear.load(is);
		binaryparam.load(is);
		labelAlpha.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */
