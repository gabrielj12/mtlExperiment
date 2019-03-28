from prettytable import PrettyTable

def resultsToFile(file,metrics_list, last_metric = False):
    with open(file,"a+") as f:
        if not last_metric:
            first_element = metrics_list[0]
            header = ["id_fold"] + list(first_element[1].keys())
            t = PrettyTable(header)
            for i in metrics_list:
                fold_id = i[0]
                metrics = i[1]
                metrics_values = [fold_id] + list(metrics.values())
                t.add_row(metrics_values)
                #print (metrics_values)

                #f.write("                 ".join(map(str,metrics_values)))
                #.write("\n")
                #print (fold_id)
                #print (metrics)
            #print (t)
            f.write(t.get_string())
            f.write("\n")
        else:
            #print (metrics_list)
            header = list(metrics_list.keys())
            metrics_values =  list(metrics_list.values())
            t = PrettyTable(header)
            t.add_row(metrics_values)
            f.write(t.get_string())
            f.write("\n")