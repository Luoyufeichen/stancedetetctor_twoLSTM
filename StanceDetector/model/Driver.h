/*
 * Driver.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"

//A native neural network classfier using only word embeddings

class Driver{
	public:
		Driver(int memsize) {

		}

		~Driver() {

		}

	public:
		vector<GraphBuilder> _builders;
		ModelParams _modelparams;  // model parameters
		HyperParams _hyperparams;

		Metric _eval;
		ModelUpdate _ada;  // model update


	public:
		//embeddings are initialized before this separately.
		inline void initial() {
			if (!_hyperparams.bValid()){
				std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
				return;
			}
			if (!_modelparams.initial(_hyperparams)){
				std::cout << "model parameter initialization Error, Please check!" << std::endl;
				return;
			}
			_modelparams.exportModelParams(_ada);
			//_modelparams.exportCheckGradParams(_checkgrad);

			_hyperparams.print();

			setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		}


		/*inline void TestInitial() {
			if (!_hyperparams.bValid()) {
				std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
				return;
			}
			if (!_modelparams.TestInitial(_hyperparams)) {
				std::cout << "model parameter initialization Error, Please check!" << std::endl;
				return;
			}
			_modelparams.exportModelParams(_ada);
			//_modelparams.exportCheckGradParams(_checkgrad);

			_hyperparams.print();

			_builders.resize(_hyperparams.batch);

			for (int idx = 0; idx < _hyperparams.batch; idx++) {
				_builders[idx].createNodes(GraphBuilder::max_sentence_length);
				_builders[idx].initial(&_cg, _modelparams, _hyperparams);
			}

			setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		}*/



		inline dtype train(const vector<Example>& examples, int iter) {
			_eval.reset();
			int example_num = examples.size();

			Graph graph;
			_builders.clear();
			_builders.resize(_hyperparams.batch);   

			for (int idx = 0; idx < _hyperparams.batch; idx++) {
				_builders.at(idx).createNodes(GraphBuilder::max_sentence_length);
				_builders.at(idx).initial(&graph, _modelparams, _hyperparams);
			}

			if (example_num > _builders.size()) {
				std::cout << "input example number larger than predefined batch number" << std::endl;
				return 1000;
			}
			
			dtype cost = 0.0;
			
			for (int count = 0; count < example_num; count++) {
				const Example& example = examples[count];
	
				//forward
				
				_builders[count].forward(example.m_feature, true);

			}
			graph.compute();

			for (int count = 0; count < example_num; count++) {
				const Example& example = examples[count];

				cost += _modelparams.loss.loss(&_builders[count]._neural_output_all, example.m_label, _eval, example_num);
			}

			graph.backward();

			if (_eval.getAccuracy() < 0) {
				std::cout << "strange" << std::endl;
			}

			return cost;
		}

		inline void predict(const Feature& feature, int& result) {
			Graph graph;
			GraphBuilder builder;
			builder.createNodes(GraphBuilder::max_sentence_length);
			builder.initial(&graph, _modelparams, _hyperparams);
			builder.forward(feature);
			graph.compute();
			bool bTargetInTweet = IsTargetIntweet(feature);
			_modelparams.loss.predict(&builder._neural_output_all, result, bTargetInTweet);
		}

		inline bool IsTargetIntweet(const Feature& feature) {
			string words = "";
			for (int i = 0; i < feature.m_words.size(); i++)
				words = words + feature.m_words[i];
			string::size_type idx;
			if (feature.m_target[0] == "hillary") {
				idx = words.find("hillary");
				if (idx != string::npos) return true;
				idx = words.find("clinton");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "trump") {
				idx = words.find("trump");
				if (idx != string::npos) return true;
				idx = words.find("donald");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "climate") {
				idx = words.find("climate");
				if (idx != string::npos) return true;
			}
			if (feature.m_target[0] == "feminism") {
				idx = words.find("feminism");
				if (idx != string::npos) return true;
				idx = words.find("feminist");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "abortion") {
				idx = words.find("abortion");
				if (idx != string::npos) return true;
				idx = words.find("aborting");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "atheism") {
				idx = words.find("atheism");
				if (idx != string::npos) return true;
				idx = words.find("atheist");
				if (idx != string::npos) return true;

			}
			return false;
		}


		void updateModel() {
			//_ada.update();
			//_ada.update(5.0);
			//_ada.update(10);
			_ada.updateAdam(10);
		}

		void checkgrad(const vector<Example>& examples, int iter){
			ostringstream out;
			out << "Iteration: " << iter;
		}




	private:
		inline void resetEval() {
			_eval.reset();
		}


		inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
			_ada._alpha = adaAlpha;
			_ada._eps = adaEps;
			_ada._reg = nnRegular;
		}

};

#endif /* SRC_Driver_H_ */
