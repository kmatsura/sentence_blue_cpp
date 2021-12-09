#include <map>
#include <unordered_map>
#include <vector>
#include <cmath>

using namespace std;
typedef typename std::vector<std::string> WordVector;

float modifiedPrecision(WordVector reference_text, WordVector hypothesis_text, int n);
float brevityPenalty(int ref_length, int hyp_length);
string getNgramText(WordVector test_vector, int s, int n);

/**
 * @brief Calculate the BLEU score between sentences.
 * @param weights weight vector
 * @param smooth Whether to perform smoothing when the match rate is zero.
 * @return BLEU Score
 */
float sentenceBLEU(WordVector reference_text, WordVector hypothesis_text, vector<float> weights = {0.25, 0.25, 0.25, 0.25}, bool smooth = false)
{
    int hyp_length = hypothesis_text.size();
    int ref_length = reference_text.size();
    if (hyp_length < weights.size() || ref_length < weights.size()) {
        int length = min(hyp_length, ref_length);
        float w = 1.0 / length;
        vector<float> new_weight(length, w);
        return sentenceBLEU(reference_text, hypothesis_text, new_weight, smooth);
    }
    unordered_map<int, int> p_numerators;
    unordered_map<int, int> p_denominators;
    for (int i = 1; i <= weights.size(); i++)
    { // initialize
        p_numerators[i] = 0;
        p_denominators[i] = max(1, hyp_length - i + 1);
    }
    for (int i = 1; i <= weights.size(); i++)
    {
        float p_i_numerator = modifiedPrecision(reference_text, hypothesis_text, i);
        if (p_i_numerator == 0)
        {
            if (i == 1) {
                return 0;
            }
            break; // If the number of matches in n-grams is 0, then 0 after (n+1)-gram
        }
        p_numerators[i] = p_i_numerator;
    }
    float bp = brevityPenalty(ref_length, hyp_length);
    float s = 0;
    for (int i = 1; i <= weights.size(); i++)
    {
        float p_i = 0;
        if (smooth && p_numerators[i] == 0)
        {
            p_i = (p_numerators[i] + 0.1) / p_denominators[i];
        }
        else
        {
            p_i = double(p_numerators[i]) / p_denominators[i];
        }
        s += weights[i-1] * log(p_i);
    }
    s = bp * exp(s);
    return s;
}

/**
 * @brief Count the number of n-gram matches.
 * @param n n of n-gram
 * @return Numerator of the match rate
 */
float modifiedPrecision(WordVector reference_text, WordVector hypothesis_text, int n)
{
    unordered_map<string, int> hyp_counter;
    unordered_map<string, int> ref_counter;
    float p_i_numerator;
    for (int i = 0; i < hypothesis_text.size() - n+1; i++)
    {
        string key = getNgramText(hypothesis_text, i, n); // ちょっと怪しい実装
        if (hyp_counter.count(key) == 0)
        {
            hyp_counter[key] = 1;
        }
        else
        {
            hyp_counter[key] += 1;
        }
    }
    for (int i = 0; i < reference_text.size() - n+1; i++)
    {
        string key = getNgramText(reference_text, i, n);
        if (ref_counter.count(key) == 0)
        {
            ref_counter[key] = 1;
        }
        else
        {
            ref_counter[key] += 1;
        }
    }
    int same_count = 0;
    for (auto pair : hyp_counter)
    {
        string key = pair.first;
        if (ref_counter.count(key) == 0)
        {
            continue;
        }
        same_count += min(pair.second, ref_counter[key]);
    }
    p_i_numerator = same_count;
    return p_i_numerator;
}

/**
 * @brief Penalize short sentences.
 * @param n n of n-gram
 * @return value of penalty
 */
float brevityPenalty(int ref_length, int hyp_length)
{
    if (hyp_length > ref_length)
    {
        return 1;
    }
    else if (hyp_length == 0)
    {
        return 0;
    }
    else
    {
        return exp(1 - float(ref_length) / hyp_length);
    }
}

/**
 * @brief Return n-grams in string format. Connect words with "-".
 * @param s start pos
 * @param n n of n-gram
 * @return n-gram
 */
string getNgramText(WordVector test_vector, int s, int n)
{
    string ngram_text = "";
    for (int i = 0; i < n - 1; i++)
    {
        ngram_text += test_vector[s + i] + "-";
    }
    ngram_text += test_vector[s + n - 1];
    return ngram_text;
}