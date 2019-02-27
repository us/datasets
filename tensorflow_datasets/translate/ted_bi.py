# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TED talk bilingual data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow_datasets.core import api_utils
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Bilingual data sets derived from TED talk transcripts.
"""

_CITATION = """\
@InProceedings{qi-EtAl:2018:N18-2,
  author    = {Qi, Ye  and  Sachan, Devendra  and  Felix, Matthieu  and  Padmanabhan, Sarguna  and  Neubig, Graham},
  title     = {When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  pages     = {529--535},
  abstract  = {The performance of Neural Machine Translation (NMT) systems often suffers in low-resource scenarios where sufficiently large-scale parallel corpora cannot be obtained. Pre-trained word embeddings have proven to be invaluable for improving performance in natural language analysis tasks, which often suffer from paucity of data. However, their utility for NMT has not been extensively explored. In this work, we perform five sets of experiments that analyze when we can expect pre-trained word embeddings to help in NMT tasks. We show that such embeddings can be surprisingly effective in some cases -- providing gains of up to 20 BLEU points in the most favorable setting.},
  url       = {http://www.aclweb.org/anthology/N18-2084}
}
"""

_DATA_URL = "http://www.phontron.com/data/qi18naacl-dataset.tar.gz"

_VALID_LANGUAGE_PAIRS = (
    ("az", "en"),
    ("be", "en"),
    ("gl", "en"),
    ("pt", "en"),
    ("ru", "en"),
    ("tr", "en"),
    ("az_tr", "en"),
    ("be_ru", "en"),
    ("gl_pt", "en"),
    ("es", "pt"),
    ("fr", "pt"),
    ("it", "pt"),
    ("ru", "pt"),
    ("he", "pt"),
)


class TedBiConfig(tfds.core.BuilderConfig):
  """BuilderConfig for TED talk bilingual data."""

  @api_utils.disallow_positional_args
  def __init__(self, language_pair=(None, None), **kwargs):
    """BuilderConfig for TED talk bilingual data.

    Args:
      language_pair: pair of languages that will be used for translation. The
        first language should either be a 2-letter coded string or two such
        strings joined by an underscore (e.g., "az" or "az_tr"). In cases where
        it contains two languages, the train data set will contain an
        (unlabelled) mix of the two languages and the validation and test sets
        will contain only the first language. This dataset will refer to the
        source language by the 5-letter string with the underscore. The second
        language in `language_pair` must be a 2-letter coded string. First will
        be used as source and second as target in supervised mode.
      **kwargs: keyword arguments forwarded to super.
    """
    name = "%s_to_%s" % (language_pair[0].replace("_", ""), language_pair[1])

    description = ("Translation dataset from %s to %s in plain text.") % (
        language_pair[0], language_pair[1])
    super(TedBiConfig, self).__init__(
        name=name, description=description, **kwargs)

    # Validate language pair.
    assert language_pair in _VALID_LANGUAGE_PAIRS, (
        "Config language pair (%s, "
        "%s) not supported") % language_pair

    self.language_pair = language_pair


class TedBiTranslate(tfds.core.GeneratorBasedBuilder):
  """TED talk bilingual data set."""

  BUILDER_CONFIGS = [
      TedBiConfig(language_pair=pair, version="0.0.1")
      for pair in _VALID_LANGUAGE_PAIRS
  ]

  def _info(self):
    source, target = self.builder_config.language_pair
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.TranslationVariableLanguages(),
        supervised_keys=(source, target),
        urls=["https://github.com/neulab/word-embeddings-for-nmt"],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    dl_dir = dl_manager.download_and_extract(_DATA_URL)
    source, target = self.builder_config.language_pair

    data_dir = os.path.join(dl_dir, "datasets", "%s_to_%s" % (source, target))

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={
                "source_file":
                    os.path.join(data_dir, "{}.train".format(
                        source.replace("_", "-"))),
                "target_file":
                    os.path.join(data_dir, "{}.train".format(target))
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=1,
            gen_kwargs={
                "source_file":
                    os.path.join(data_dir, "{}.dev".format(
                        source.split("_")[0])),
                "target_file":
                    os.path.join(data_dir, "{}.dev".format(target))
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,
            gen_kwargs={
                "source_file":
                    os.path.join(data_dir, "{}.test".format(
                        source.split("_")[0])),
                "target_file":
                    os.path.join(data_dir, "{}.test".format(target))
            }),
    ]

  def _generate_examples(self, source_file, target_file):
    """This function returns the examples in the raw (text) form."""
    with tf.io.gfile.GFile(source_file) as f:
      source_sentences = f.read().split("\n")
    with tf.io.gfile.GFile(target_file) as f:
      target_sentences = f.read().split("\n")

    assert len(target_sentences) == len(
        source_sentences), "Sizes do not match: %d vs %d for %s vs %s." % (len(
            source_sentences), len(target_sentences), source_file, target_file)

    source, target = self.builder_config.language_pair
    for l1, l2 in zip(source_sentences, target_sentences):
      result = {source: l1, target: l2}
      # Make sure that both translations are non-empty.
      if all(result.values()):
        yield result
