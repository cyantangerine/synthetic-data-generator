"""DataTransformer module."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type

import concurrent
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import autonotebook as tqdm

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import CategoricalEncoderType, Metadata, ContinuousEncoderType
from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader
from sdgx.models.components.optimize.sdv_ctgan.types import (
    ActivationFuncType,
    CategoricalEncoderInstanceType,
    ColumnTransformInfo,
    SpanInfo,
)
from sdgx.models.components.sdv_rdt.transformers import (
    ClusterBasedNormalizer,
    NormalizedFrequencyEncoder,
    NormalizedLabelEncoder,
    OneHotEncoder,
)
from sdgx.models.components.sdv_rdt.transformers.numerical import DirectNormalizer
from sdgx.utils import logger
from sdgx.log import USER_DEFINED_LOG_LEVEL

CategoricalEncoderParams = NamedTuple(
    "CategoricalEncoderParams",
    (
        ("encoder", Callable[[], CategoricalEncoderInstanceType]),
        ("categories_caculator", Callable[[CategoricalEncoderInstanceType], int]),
        ("activate_fn", ActivationFuncType),
    ),
)
CategoricalEncoderMapper: Dict[CategoricalEncoderType, CategoricalEncoderParams] = {
    CategoricalEncoderType.ONEHOT: CategoricalEncoderParams(
        lambda: OneHotEncoder(), lambda encoder: len(encoder.dummies), ActivationFuncType.SOFTMAX
    ),
    CategoricalEncoderType.LABEL: CategoricalEncoderParams(
        lambda: NormalizedLabelEncoder(order_by="alphabetical"),
        lambda encoder: 1,
        ActivationFuncType.LINEAR,
    ),
    CategoricalEncoderType.FREQUENCY: CategoricalEncoderParams(
        lambda: NormalizedFrequencyEncoder(), lambda encoder: 1, ActivationFuncType.LINEAR
    ),
}


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005, metadata=None):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self.metadata: Metadata = metadata
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_categorical_encoder(
        self, column_name: str, data: pd.DataFrame, encoder_type: CategoricalEncoderType
    ) -> Tuple[CategoricalEncoderInstanceType, int, ActivationFuncType]:
        if encoder_type not in CategoricalEncoderMapper.keys():
            raise ValueError("Unsupported encoder type {0}.".format(encoder_type))
        p: CategoricalEncoderParams = CategoricalEncoderMapper[encoder_type]
        encoder = p.encoder()
        encoder.fit(data, column_name)
        num_categories = p.categories_caculator(encoder)
        # Notice: if `activate_fn` is modified, the function `is_onehot_encoding_column` in `DataSampler` should also be modified.
        activate_fn = p.activate_fn
        return encoder, num_categories, activate_fn

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        encoder_type = None
        if self.metadata is None or not self.metadata.continuous_encoder:
            encoder_type = ContinuousEncoderType.GMM
        else:
            if column_name in self.metadata.continuous_encoder:
                encoder_type = self.metadata.continuous_encoder[column_name]
            else:
                encoder_type = ContinuousEncoderType.GMM

        num_components = None
        output_dim = None
        output_info = []
        if encoder_type == ContinuousEncoderType.GMM:
            encoder = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))
            encoder.fit(data, column_name)
            num_components = sum(encoder.valid_component_indicator)
            output_info = [SpanInfo(1, "tanh"), SpanInfo(num_components, "softmax")]
            output_dim = num_components + 1
        elif encoder_type == ContinuousEncoderType.NONE:
            encoder = DirectNormalizer()
            encoder.fit(data, column_name)
            num_components = 1
            output_dim = 1
            output_info = [SpanInfo(1, "linear")]
        else:
            raise Exception(f"Unknown encoder type {encoder_type} for column {column_name}.")

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=encoder,
            output_info=output_info,
            output_dimensions=output_dim,
        )

    def _fit_discrete(self, data, encoder_type: CategoricalEncoderType = None):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        encoder, activate_fn, selected_encoder_type = None, None, None
        column_name = data.columns[0]

        # Load encoder from metadata
        if encoder_type is None and self.metadata:
            selected_encoder_type = encoder_type = self.metadata.get_column_encoder_by_name(
                column_name
            )
        # if the encoder is not be specified, using onehot.
        if encoder_type is None:
            encoder_type = "onehot"
        # if the encoder is onehot, or not be specified.
        num_categories = -1  # if zero may cause crash to onehot.
        if encoder_type == "onehot":
            encoder, num_categories, activate_fn = self._fit_categorical_encoder(
                column_name, data, encoder_type
            )

        # if selected_encoder_type is not specified and using onehot num_categories > threshold, change the encoder.
        if not selected_encoder_type and self.metadata and num_categories != -1:
            encoder_type = (
                self.metadata.get_column_encoder_by_categorical_threshold(num_categories)
                or encoder_type
            )

        if encoder_type == "onehot":
            pass
        else:
            encoder, num_categories, activate_fn = self._fit_categorical_encoder(
                column_name, data, encoder_type
            )

        assert encoder and activate_fn

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=encoder,
            output_info=[SpanInfo(num_categories, activate_fn)],
            # Notice: if `output_info` is modified, the function `is_onehot_encoding_column` in `DataSampler` should also be modified.
            output_dimensions=num_categories,
        )

    def fit(self, data_loader: DataLoader, discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list: List[List[SpanInfo]] = []
        self.output_dimensions: int = 0
        self.dataframe: bool = True

        self._column_raw_dtypes = data_loader[: data_loader.chunksize].infer_objects().dtypes
        self._column_transform_info_list: List[ColumnTransformInfo] = []
        
        # # 并行化处理
        def process_column(column_name, data_loader, discrete_columns, logger):
            if column_name in discrete_columns:
                #  or column_name in self.metadata.label_columns
                logger.debug(f"Fitting discrete column {column_name}...")
                column_transform_info = self._fit_discrete(data_loader[[column_name]])
            else:
                logger.debug(f"Fitting continuous column {column_name}...")
                column_transform_info = self._fit_continuous(data_loader[[column_name]])
            return column_transform_info
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = {executor.submit(process_column, column_name, data_loader, discrete_columns, logger): column_name
        #             for column_name in data_loader.columns()}
            
        #     for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preparing data", delay=3):
        #         column_transform_info = future.result()
        #         self.output_info_list.append(column_transform_info.output_info)
        #         self.output_dimensions += column_transform_info.output_dimensions
        #         self._column_transform_info_list.append(column_transform_info)
        if len(data_loader.columns()) > 100:
            processes = []
            for column_name in data_loader.columns():
                process = delayed(process_column)(column_name, data_loader, discrete_columns, logger)
                processes.append(process)
            
            p = Parallel(n_jobs=-1, return_as="generator")
            for column_transform_info in tqdm.tqdm(
                p(processes), desc="Preparing data", total=len(processes), delay=3
            ):
                self.output_info_list.append(column_transform_info.output_info)
                self.output_dimensions += column_transform_info.output_dimensions
                self._column_transform_info_list.append(column_transform_info)
        else:
            for column_name in tqdm.tqdm(data_loader.columns(), desc="Preparing data", delay=3):
                column_transform_info = process_column(column_name, data_loader, discrete_columns, logger)
                self.output_info_list.append(column_transform_info.output_info)
                self.output_dimensions += column_transform_info.output_dimensions
                self._column_transform_info_list.append(column_transform_info)
        if USER_DEFINED_LOG_LEVEL == 'INFO' or USER_DEFINED_LOG_LEVEL == 'DEBUG':
            log_data = [item.to_str_log() for item in self._column_transform_info_list]
            logger.info(f"Transform fit result:\n" + '\n'.join(log_data))

    def _transform_continuous(self, column_transform_info, data):
        logger.debug(f"Transforming continuous column {column_transform_info.column_name}...")
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        if isinstance(gm, ClusterBasedNormalizer):
            index = transformed[f"{column_name}.component"].to_numpy().astype(int)
            output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        logger.debug(f"Transforming discrete column {column_transform_info.column_name}...")
        encoder = column_transform_info.transform
        return encoder.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list) -> NDArrayLoader:
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        loader = NDArrayLoader.get_auto_save(raw_data)
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == "continuous":
                loader.store(self._transform_continuous(column_transform_info, data).astype(float))
            else:
                loader.store(self._transform_discrete(column_transform_info, data).astype(float))

        return loader

    def _parallel_transform(self, raw_data, column_transform_info_list) -> NDArrayLoader:
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == "continuous":
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        p = Parallel(n_jobs=-1, return_as="generator")

        loader = NDArrayLoader.get_auto_save(raw_data)
        for ndarray in tqdm.tqdm(
            p(processes), desc="Transforming data", total=len(processes), delay=3
        ):
            loader.store(ndarray.astype(float))
        return loader

    def transform(self, dataloader: DataLoader) -> NDArrayLoader:
        """Take raw data and output a matrix data."""

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if dataloader.shape[0] < 500:
            loader = self._synchronous_transform(dataloader, self._column_transform_info_list)
        else:
            loader = self._parallel_transform(dataloader, self._column_transform_info_list)

        return loader

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        if isinstance(gm, ClusterBasedNormalizer):
            data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
            data = data.astype(float)
            data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
            if sigmas is not None:
                selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
                data.iloc[:, 0] = selected_normalized_value
        else:
            data = pd.DataFrame(column_data[:, 0], columns=list(gm.get_output_sdtypes()))
        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        encoder = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(encoder.get_output_sdtypes()))
        return encoder.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        def process_inv_transform(column_transform_info, st):
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]
            if column_transform_info.column_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )
            return recovered_column_data
        
        
        # TODO using pd.df.apply to increase performance.
        st = 0
        recovered_column_data_list = []
        column_names = []
        if len(self._column_transform_info_list) < 100:
            for column_transform_info in tqdm.tqdm(
                self._column_transform_info_list, desc="Inverse transforming", delay=3
            ):
                dim = column_transform_info.output_dimensions
                recovered_column_data = process_inv_transform(column_transform_info, st)
                recovered_column_data_list.append(recovered_column_data)
                column_names.append(column_transform_info.column_name)
                st += dim
        else:
            processes = []
            for column_transform_info in self._column_transform_info_list:
                dim = column_transform_info.output_dimensions
                column_names.append(column_transform_info.column_name)
                process = delayed(process_inv_transform)(column_transform_info, st)
                processes.append(process)
                st += dim
                
            p = Parallel(n_jobs=-1, return_as="generator")
            for recovered_column_data in tqdm.tqdm(
                p(processes), desc="Inverse transforming", delay=3, total=len(processes)
            ):
                recovered_column_data_list.append(recovered_column_data)
                
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot),
        }
