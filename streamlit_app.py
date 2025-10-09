# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
import argparse
import html
import json
import os
import pdb
import random
import sys

import numpy as np

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

random.seed(42)


def load_sampled_data(args):
    """Function to load JSON data."""
    save_path = os.path.join(
        args.manifold_dir,
        # args.sampled_data_path,
        f"N={args.sample_size}_num_neg={args.num_neg_pssgs}_{args.data_sample_method}.json",
    )
    if os.path.exists(save_path) and not args.overwrite:
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    final_sampled_instances = []
    for query_type in ["explicit", "implicit"]:
        total_sample_size = args.sample_size // 2

        data_path = os.path.join(
            args.manifold_dir, args.data_path.format(query_type=query_type)
        )
        df = pd.read_parquet(data_path)
        grouped = df.groupby(["pid", "sub_qid"])

        instances = []

        for (pid, sub_qid), group_df in grouped:
            query = group_df["query"].iloc[0]
            answers = group_df["answers"].iloc[0].tolist()
            passages = group_df["passage"].tolist()[: args.num_neg_pssgs + 1]
            labels = group_df["label"].tolist()[: args.num_neg_pssgs + 1]
            if labels[0] != 1:
                pdb.set_trace()
            original_ids = list(range(len(passages)))
            combined = list(zip(passages, labels, original_ids))
            random.shuffle(combined)
            passages_shuffled, labels_shuffled, shuffled_original_ids = zip(*combined)
            passages = list(passages_shuffled)
            labels = list(labels_shuffled)
            passage_ids = list(shuffled_original_ids)

            instance = {
                "pid": pid,
                "sub_qid": sub_qid,
                "query": query,
                "answers": answers,
                "candidate_passages": passages,
                "passage_ids": passage_ids,
                "labels": labels,
            }
            instances.append(instance)

        instances_df = pd.DataFrame(instances)

        if args.data_sample_method == "random":
            sampled_df = instances_df.sample(n=total_sample_size, random_state=42)
        else:
            groups = instances_df.groupby("pid")
            n_groups = len(groups)
            n_per_group = int(np.ceil(total_sample_size / n_groups))

            sampled_df = groups.apply(
                lambda x: x.sample(n=min(len(x), n_per_group), random_state=42)
            ).reset_index(drop=True)

            remaining = total_sample_size - len(sampled_df)
            if remaining > 0:
                remaining_df = instances_df.loc[
                    ~instances_df.index.isin(sampled_df.index)
                ]
                if len(remaining_df) > 0:
                    additional_sample = remaining_df.sample(
                        n=min(remaining, len(remaining_df)), random_state=42
                    )
                    sampled_df = pd.concat(
                        [sampled_df, additional_sample], ignore_index=True
                    )

        if len(sampled_df) > total_sample_size:
            final_sampled_df = sampled_df.sample(n=total_sample_size, random_state=42)
        else:
            final_sampled_df = sampled_df

        sampled_instances = final_sampled_df.to_dict("records")
        final_sampled_instances.extend(sampled_instances)

    cnt = 0
    for inst in final_sampled_instances:
        inst["instance_id"] = cnt
        cnt += 1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_sampled_instances, f, indent=4)

    return final_sampled_instances


def save_annotations():
    """Converts annotations stored in the session state into a DataFrame for download."""
    records = []

    for instance_id, selection_info in st.session_state.annotations.items():
        records.append(
            {
                "instance_id": instance_id,
                "selected_passage_number": selection_info["number"],
                "selected_passage_text": selection_info["text"],
                # --- MODIFICATION START ---
                # 코멘트 정보를 추가합니다. .get()을 사용해 코멘트가 없는 경우에도 에러 없이 빈 문자열을 반환합니다.
                "comment": selection_info.get("comment", ""),
                # --- MODIFICATION END ---
                "displayed_passage_ids": selection_info["passage_ids"],
                "displayed_labels": selection_info["labels"],
            }
        )

    if not records:
        return pd.DataFrame().to_csv(index=False).encode("utf-8")

    df = pd.DataFrame(records)
    return df.to_csv(index=False).encode("utf-8")


def main(args: argparse.Namespace):
    """Main function to run the Streamlit annotation app."""
    st.set_page_config(layout="wide")

    def scroll_to_top():
        components.html(
            """
            <script>
                window.location.href = '#top';
            </script>
            """,
            height=0,
        )

    # Initialize session state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
        st.session_state.annotations = {}

    if st.session_state.get("scroll_to_top", False):
        scroll_to_top()
        st.session_state.scroll_to_top = False

    data = load_sampled_data(args)[args.start : args.end]
    total_items = len(data)

    st.title("Human Annotation")
    st.markdown("<a id='top'></a>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Progress")
        st.progress((st.session_state.current_index + 1) / total_items)
        st.metric(
            label="Completion",
            value=f"{st.session_state.current_index + 1} / {total_items}",
        )
        st.header("Save Results")
        st.download_button(
            label="Download Annotations (CSV)",
            data=save_annotations(),
            file_name=f"annotations_{args.username}.csv",
            mime="text/csv",
        )

    current_item = data[st.session_state.current_index]
    instance_id = current_item["instance_id"]

    st.header(f"Instance #{instance_id+1}")
    st.subheader("Query")
    st.info(current_item["query"])

    st.markdown("---")
    st.subheader("Candidate Passages")
    st.write(
        "From the candidate passages below, please select the **single best passage** that serves as the most accurate and up-to-date evidence, considering the query's timestamp."
    )

    passages = current_item["candidate_passages"]
    st.session_state.current_instance_id = instance_id
    st.session_state.current_passages = passages
    st.session_state.current_passage_ids = current_item["passage_ids"]
    st.session_state.current_labels = current_item["labels"]

    # --- MODIFICATION START ---
    # 주석과 선택을 모두 업데이트하는 통합 함수
    def update_annotation():
        instance_id = st.session_state.current_instance_id
        passages = st.session_state.current_passages
        passage_ids = st.session_state.current_passage_ids
        labels = st.session_state.current_labels

        # session_state에서 선택과 코멘트를 안전하게 가져옵니다.
        selection = st.session_state.get(f"selection_{instance_id}")
        comment_text = st.session_state.get(f"comment_{instance_id}", "")

        # 아직 아무것도 선택하지 않았다면, 아무 작업도 하지 않습니다.
        if selection is None:
            return

        # 저장할 기본 데이터 구조
        annotation_data = {
            "passage_ids": passage_ids,
            "labels": labels,
            "comment": comment_text,
        }

        if selection == "None of the above":
            annotation_data["number"] = "None"
            annotation_data["text"] = "None"
        else:
            try:
                selected_number = int(selection.split(" ")[1])
                selected_index = selected_number - 1
                selected_passage_text = passages[selected_index]
                annotation_data["number"] = selected_number
                annotation_data["text"] = selected_passage_text
            except (ValueError, IndexError):
                # 잘못된 선택 값인 경우, 업데이트하지 않습니다.
                return

        # session_state에 최종 주석 정보를 저장합니다.
        st.session_state.annotations[str(instance_id)] = annotation_data

    # --- MODIFICATION END ---

    for i, passage_text in enumerate(passages, 1):
        with st.container(border=True):
            st.markdown(f"### Passage [{i}]")
            st.markdown(
                f"""
                <div style="
                    background-color: #F0F2F6;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    font-family: monospace;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                ">
                {html.escape(passage_text)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("---")

    radio_options = [f"Passage {i}" for i in range(1, len(passages) + 1)]
    radio_options.append("None of the above")

    st.radio(
        label="**DECISION**: Select the single best passage:",
        options=radio_options,
        key=f"selection_{instance_id}",
        index=None,
        horizontal=True,
        on_change=update_annotation,
    )

    # --- MODIFICATION START ---
    # 코멘트를 입력받기 위한 텍스트 영역을 추가합니다.
    st.text_area(
        label="Comment (optional):",
        key=f"comment_{instance_id}",
        on_change=update_annotation,
        help="You can add any notes or justifications for your choice here.",
    )
    # --- MODIFICATION END ---

    # --- Navigation Buttons ---
    # 1. 경고 메시지를 표시할 공간을 미리 만듭니다.
    alert_placeholder = st.empty()

    col_nav1, col_nav2, col_nav3 = st.columns([1, 8, 1])
    with col_nav1:
        if st.button(
            "Previous",
            use_container_width=True,
            disabled=(st.session_state.current_index == 0),
        ):
            st.session_state.current_index -= 1
            st.session_state.scroll_to_top = True
            st.rerun()

    with col_nav3:
        if st.button(
            "Next",
            use_container_width=True,
            disabled=(st.session_state.current_index >= total_items - 1),
        ):
            # 2. 'Next'를 누르면 현재 질문에 대한 답변이 있는지 확인합니다.
            selection_key = f"selection_{instance_id}"

            # .get()을 사용하여 키가 없어도 에러가 나지 않도록 합니다.
            if st.session_state.get(selection_key) is None:
                # 답변이 없으면 경고 메시지를 표시합니다.
                alert_placeholder.warning(
                    "Please select an answer. You must choose one option to proceed.",
                    icon="⚠️",
                )
            else:
                # 답변이 있으면 다음 항목으로 넘어갑니다.
                st.session_state.current_index += 1
                st.session_state.scroll_to_top = True
                st.rerun()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--manifold_dir", type=str, default="./")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument(
        "--data_path",
        type=str,
        default="tree/sohyun/TvWiki-Rank/output/preprocess_v5.2/{query_type}.parquet",
    )
    parser.add_argument(
        "--data_sample_method",
        type=str,
        default="random",
        choices=["random", "group_by_pid"],
    )
    # parser.add_argument(
    #     "--sampled_data_path",
    #     type=str,
    #     default="tree/sohyun/TvWiki-Rank/human_eval/sampled_data/v1",
    # )
    parser.add_argument("--num_neg_pssgs", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--username", type=str, default="hayeon")
    parser.add_argument("--overwrite", type=int, default=0)
    return parser.parse_args()


# This block ensures that the main() function is called only when the script is executed directly
if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    if args.debug:
        args.sample_size = 9
        args.username = "debug"
        args.sampled_data_path = "tree/sohyun/TvWiki-Rank/human_eval/sampled_data/debug"
        args.start = 3
        args.end = 6
    main(args)
    # streamlit run annotate_app.py
    # streamlit run annotate_app.py --server.address 0.0.0.0