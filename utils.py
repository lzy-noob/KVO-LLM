from typing import Optional
from prettytable import PrettyTable



def print_results(
    eval_res: dict, # evaluation results on selected benchmarks
    model: str, # model name
    custom_flag: Optional[str] # predefined flags for the current test
):
    """
        eval_res: dict() ->
            {
                "lm_eval_harness": {
                    "arc_easy": float (results),
                    ...
                },
                "ppl_eval": {
                    "wikitext": float (results),
                    ...
                }
                "longbench": {
                    "triviaqa": float (results),
                    ...
                }
            }
    """
    for bench_key in eval_res.keys():
        bench_table = PrettyTable()
        bench_table.title = f"{model} on {bench_key}"
        if custom_flag is not None:
            bench_table.title += f" for {custom_flag}"
        bench_table.field_names = ["task",
                                   "perplexity" if bench_key is "ppl_eval" else "accuracy"]
        for task in eval_res[bench_key].keys():
            bench_table.add_row([task, eval_res[bench_key][task]])
        print(bench_table)
        print()

