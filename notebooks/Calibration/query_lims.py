import requests
import json
import pathlib as pl

# These are URLs for LIMS data.
# You can just paste this into your browser too e.g. "http://lims2/donors/info/details.json?external_donor_name=672102&parent_specimens=true"
donor_info_url = "http://lims2/donors/info/details.json?external_donor_name={donor_name}&parent_specimens=true"
weight_records_url = "http://lims2/visual_behavior_details?donor_id={donor_id}"

mice = [672102, 672013, 672104, 672105, 672106, 672107]
timeout = None


def lims_request(url, timeout=None):
    response = requests.get(url, timeout=timeout)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None


for mouse_name in mice:
    url = donor_info_url.format(donor_name=mouse_name)
    print(url)
    donor_info = lims_request(url)

    if donor_info is not None:
        file = pl.Path(__file__).parent / f"info_{mouse_name}.json"
        with open(file, "w") as f:
            f.write(json.dumps(donor_info))

        donor_id = donor_info[0]["id"]
        print(donor_id)

        url_weights = weight_records_url.format(donor_id=donor_id)
        weight_records = lims_request(url_weights)
        if weight_records is not None:
            print(len(weight_records))

            file = pl.Path(__file__).parent / f"weights_{mouse_name}.json"
            with open(file, "w") as f:
                f.write(json.dumps(weight_records))

    else:
        print(f"No records found for mouse {mouse_name}")
