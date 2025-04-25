from collections import defaultdict
import numpy as np
from .scorers import average_scorer, naive_scorer, softmax_scorer

def condense_rankings(data: dict) -> list:
    """
    Combine ranked list items with the same label values into a single object.

    This function iterates over a list of response items, where each response
    item contains an input ID and a list of rankings. For each ranking item,
    it consolidates items with the same label by aggregating their descriptions,
    distances, and rankings.

    Parameters
    ----------
    data (list): A list of dictionaries, where each dictionary represents a
                 response item with the following structure:
                 {
                     "input_id": <some_id>,
                     "response": [
                         {
                             "label": <label>,
                             "description": <description>,
                             "distance": <distance>,
                             "rank": <rank>
                         },
                         ...
                     ]
                 }

    Returns
    -------
    list: A list of dictionaries, where each dictionary represents a condensed
          response item with aggregated data per label. The structure is:
          {
              "input_id": <input_id as string>,
              "response": [
                  {
                      "label": <label>,
                      "descriptions": [<description1>, ...],
                      "distances": [<distance1>, ...],
                      "rankings": [<rank1>, ...]
                  },
                  ...
              ]
          }
    """

    # for each input id and response
    condensed_rankings = []
    for response_item in data:
        input_id = response_item["input_id"]
        origingal_ranking_list = response_item["response"]
        combined_dict = defaultdict(
            lambda: {
                "descriptions": [],
                "distances": [],
                "rankings": [],
            }
        )

        # for each ranking associated with an input id collect all the distances, descriptions and rankings for the same label
        for item in origingal_ranking_list:
            label = item["label"]
            combined_dict[label]["descriptions"].append(item["description"])
            combined_dict[label]["distances"].append(item["distance"])
            combined_dict[label]["rankings"].append(item["rank"])

        # combine collected items into a structure and return
        condensed_ranking = [
            {
                "label": label,
                "descriptions": data["descriptions"],
                "distances": data["distances"],
                "rankings": data["rankings"],
            }
            for label, data in combined_dict.items()
        ]
        condensed_rankings.append(
            {"input_id": str(input_id), "response": condensed_ranking}
        )

    return condensed_rankings

def create_deduplicated_response(
    initial_response: dict, scoring_method: callable
) -> list:
    """
    Transform the initial FastAPI response into a deduplicated response, combining ranking scores for the same label per input_id.

    This function utilizes `condense_rankings` to aggregate rankings with the
    same label. It then computes new scores for each label using a provided
    scoring method, sorts the results, and applies a softmax transformation
    to the scores.

    Parameters
    ----------
    initial_response (dict): A dictionary representing the initial API response,
                             containing input IDs and ranking lists.
    scoring_method (callable): A function that computes a score from a provided
                               list of numbers (e.g., distances).

    Returns
    -------
    list: A list of dictionaries, each containing an input_id and its
          associated deduplicated rankings, with updated scores and ranks.
          The structure is:
          {
              "input_id": <input_id>,
              "response": [
                  {
                      "label": <label>,
                      "distance": <calculated distance>,
                      "description": <selected description>,
                      "rank": <new rank>,
                      "score": <softmax score>
                  },
                  ...
              ]
          }
    """

    # final response holdall for new data
    deduplicated_response = []

    # pack all rankings with the same label into a single object, per iput_id
    condensed_responses = condense_rankings(initial_response)

    # for each input id and condensed response assocated with the input id
    for item in condensed_responses:
        input_id = item["input_id"]
        condensed_ranking_list = item["response"]
        deduplicated_ranking_list = []

        # for each unique label get the right score and description from the condensed ranking
        for each in condensed_ranking_list:
            distance, index = naive_scorer(each["distances"], each["rankings"])
            rank_item = {
                "label": each["label"],
                "distance": distance,
                "description": each["descriptions"][index],
            }

            deduplicated_ranking_list.append(rank_item)

        ##Sort the deduplicated ranking based on the new distance scores, per input_id
        sorted_deduplicated_ranking_list = [
            {**item, "rank": rank + 1}
            for rank, item in enumerate(
                sorted(deduplicated_ranking_list, key=lambda x: x["distance"])
            )
        ]

        # calculate the softmax of the new deduplicated ranking, per input_id
        distances = [x["distance"] for x in sorted_deduplicated_ranking_list]
        softmax_scores = softmax_scorer(distances)
        sorted_deduplicated_ranking_list = [
            {**item, "score": str(sm_score)}
            for sm_score, item in zip(
                softmax_scores, sorted_deduplicated_ranking_list
            )
        ]

        # add the input_id, deduplicated_ranking to the persisted list.
        deduplicated_response.append(
            {
                "input_id": input_id,
                "response": sorted_deduplicated_ranking_list,
            }
        )

    return deduplicated_response
