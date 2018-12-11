#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
	public:
		const static int max_sentence_length = 1024;

	public:
		// node instances
		vector<LookupNode> _word_inputs;
		vector<LookupNode> _target_inputs;

		//LSTM1Builder _lstm_left;
		//LSTM1Builder _lstm_right;

		LSTM1Builder _lstm_left_target;
		LSTM1Builder _lstm_right_target;

		LSTM1Builder _lstm_left_tweet;
		LSTM1Builder _lstm_right_tweet;

		vector<ConcatNode> _lstm_target_concat;
		vector<ConcatNode> _lstm_tweet_concat;


		MaxPoolNode _max_pooling_tweet;
		MaxPoolNode _max_pooling_target;


		UniNode _neural_output_target;
		UniNode _neural_output_tweet;


		BiNode _neural_output_all;

		Graph *_pcg_tweet;

		ModelParams *_modelParams;


	public:
		GraphBuilder(){
		}

		~GraphBuilder(){
			clear();
		}

	public:
		//allocate enough nodes 
		inline void createNodes(int sent_length){
			_word_inputs.resize(sent_length);
			_target_inputs.resize(sent_length);


			_lstm_left_target.resize(sent_length);
			_lstm_left_tweet.resize(sent_length);

			_lstm_right_target.resize(sent_length);
			_lstm_right_tweet.resize(sent_length);

			_lstm_target_concat.resize(sent_length);
			_lstm_tweet_concat.resize(sent_length);
			
		}

		inline void clear(){
			_word_inputs.clear();
			_target_inputs.clear();


			_lstm_left_target.clear();
			_lstm_left_tweet.clear();

			_lstm_right_target.clear();
			_lstm_right_tweet.clear();

			_lstm_target_concat.clear();
			_lstm_tweet_concat.clear();
		}

	public:
		inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts){
			_pcg_tweet = pcg;
			for (int idx = 0; idx < _target_inputs.size(); idx++) {
				_target_inputs[idx].setParam(&model.words);
				_target_inputs[idx].init(opts.wordDim, -1);
				_lstm_target_concat[idx].init(opts.hiddenSize * 2, -1);
			}
			for (int idx = 0; idx < _word_inputs.size(); idx++) {
				_word_inputs[idx].setParam(&model.words);
				_word_inputs[idx].init(opts.wordDim, -1);
				_lstm_tweet_concat[idx].init(opts.hiddenSize * 2, -1);
			}

			_lstm_left_target.init(&model.lstm_target_left_params, -1, true);
			_lstm_right_target.init(&model.lstm_target_right_params, -1, false);

			_lstm_left_tweet.init(&model.lstm_tweet_left_params, -1, true);
			_lstm_right_tweet.init(&model.lstm_tweet_right_params, -1, false);

			_max_pooling_tweet.init(opts.hiddenSize * 2, -1);
			_max_pooling_target.init(opts.hiddenSize * 2, -1);

			_neural_output_target.setParam(&model.olayer_linear);
			_neural_output_target.init(opts.labelSize, -1);

			_neural_output_tweet.setParam(&model.olayer_linear);
			_neural_output_tweet.init(opts.labelSize, -1);

			_neural_output_all.setParam(&model.binaryparam);
			_neural_output_all.init(opts.labelSize, -1);



			_modelParams = &model;
		}


	public:
		// some nodes may behave different during training and decode, for example, dropout
		inline void forward(const Feature& feature, bool bTrain = false){
			_pcg_tweet->train = bTrain;
			// second step: build graph
			//forward
			int words_num = feature.m_words.size();
			int target_num = feature.m_target.size();

			if (words_num > max_sentence_length)
				words_num = max_sentence_length;
			for (int i = 0; i < target_num; i++) {
				_target_inputs[i].forward(_pcg_tweet, feature.m_target[i]);
			}

			bool bTarget = true;
			PAddNode* target_cell_left;
			PAddNode* target_cell_right;
			vector<Node*> node_t = toPointers<LookupNode,Node>(_target_inputs, target_num);

			_lstm_left_target.forward(_pcg_tweet, node_t, &_modelParams->lstm_target_left_params, target_cell_left, bTarget);
			_lstm_right_target.forward(_pcg_tweet, node_t, &_modelParams->lstm_target_right_params, target_cell_right, bTarget);
			

			vector<Node*> nodes_ct = toPointers<ConcatNode, Node>(_lstm_tweet_concat, words_num);

			for (int i = 0; i < target_num; i++) {
				_lstm_target_concat[i].forward(_pcg_tweet, &_lstm_left_target._hiddens[i], &_lstm_right_target._hiddens[i]);
			}

			_max_pooling_target.forward(_pcg_tweet, nodes_ct);

			_neural_output_target.forward(_pcg_tweet, &_max_pooling_target);



			for (int i = 0; i < words_num; i++) {
				_word_inputs[i].forward(_pcg_tweet, feature.m_words[i]);
			}

			bTarget = false;
			vector<Node*> node_w = toPointers<LookupNode, Node>(_word_inputs, words_num);
			_lstm_left_tweet.forward(_pcg_tweet, node_w, &_modelParams->lstm_tweet_left_params, target_cell_left, bTarget);
			_lstm_right_tweet.forward(_pcg_tweet, node_w, &_modelParams->lstm_tweet_right_params, target_cell_right, bTarget);

			vector<Node*> nodes_c = toPointers<ConcatNode, Node>(_lstm_tweet_concat, words_num);


			for (int i = 0; i < words_num; i++) {
				_lstm_tweet_concat[i].forward(_pcg_tweet, &_lstm_left_tweet._hiddens[i], &_lstm_right_tweet._hiddens[i]);
			}

			_max_pooling_tweet.forward(_pcg_tweet, nodes_c);

			//_neural_output_tweet.forward(_pcg_tweet, &_max_pooling_tweet);

			_neural_output_all.forward(_pcg_tweet, &_max_pooling_target, &_max_pooling_tweet);
		}
};

#endif /* SRC_ComputionGraph_H_ */
