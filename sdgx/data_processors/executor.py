from joblib import delayed, Parallel
from pandas import DataFrame
from tqdm import autonotebook as tqdm

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.formatters.int import IntValueFormatter
from sdgx.data_processors.transformers.outlier import OutlierTransformer
from sdgx.utils import logger


def get_cls_name(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, type):
        return obj.__name__
    else:
        return obj.__class__.__name__


DEFAULT_ORDER_MAP = {
    "ColumnOrderTransformer": 0,

    "ConstValueTransformer": 10,
    "PositiveNegativeFilter": 10,
    "EmptyTransformer": 10,

    "ChnPiiGenerator": 20,
    "EmailGenerator": 20,
    "DatetimeFormatter": 20,

    "OutlierTransformer": 30,
    "IntValueFormatter": 30,  # Not Need if Outlier Used

    "NonValueTransformer": 40,

    "SpecificCombinationTransformer": 50,
    "FixedCombinationTransformer": 51,

}
DEFAULT_PROCESSORS_LIST = [get_cls_name(p) for p in DEFAULT_ORDER_MAP.keys()]



class DataProcessorExecutor:

    def __len__(self):
        return len(self.data_processors_list)

    def __init__(self, data_processors: list[DataProcessor], parallel_execute=False):
        self.parallel_execute = parallel_execute
        self.data_processors_list = data_processors

        processors_type_index = {}
        extend_data_processors = []
        for index, data_processor in enumerate(self.data_processors_list):
            if isinstance(data_processor, DataProcessor):
                cls_name = get_cls_name(data_processor)
                if cls_name not in processors_type_index:
                    if cls_name not in DEFAULT_PROCESSORS_LIST:
                        extend_data_processors.append([data_processor])
                        continue
                    processors_type_index[cls_name] = index
                else:
                    raise ValueError('Duplicate data processor type using.')
            else:
                raise ValueError("Item of data_processors must be of type DataProcessor")

        if get_cls_name(IntValueFormatter) in processors_type_index and get_cls_name(OutlierTransformer) in processors_type_index:
            raise ValueError("IntValueFormatter is not needed if OutlierTransformer is using.")

        order_list = []

        last_order = -1
        for cls_name, current_order in sorted(DEFAULT_ORDER_MAP.items(), key=lambda x: x[1]):
            if cls_name in processors_type_index:
                t = (processors_type_index[cls_name], current_order)
                if last_order == current_order:
                    order_list[-1].append(t)
                else:
                    last_order = current_order
                    order_list.append([t])
        order_list = [[
            self.data_processors_list[t[0]] for t in p
        ] for p in order_list]
        self.data_processors_execute_order_list = order_list + extend_data_processors
        logger.info(f"Using data processors: {[[get_cls_name(d) for d in dl] for dl in self.data_processors_execute_order_list]}")

    def _parallel_wrapper(self, ddlist, desc, func, kwargs, solver):
        result = None
        for dlist in tqdm.tqdm(ddlist, total=len(ddlist), desc=desc, delay=3):
            if self.parallel_execute and len(dlist) > 1:
                processes = []
                for d in dlist:
                    process = delayed(func)(d, **kwargs)
                    processes.append(process)

                p = Parallel(n_jobs=-1, return_as="generator")
                for result in tqdm.tqdm(
                        p(processes), desc=f"Sub {desc}", total=len(processes), delay=3, leave=False
                ):
                    if desc != 'Preparing Data':
                        # TODO
                        raise NotImplementedError("因为存在DataFrame的复制，暂无法并行处理。")
            else:
                for d in dlist:
                    result = func(d, **kwargs)
                    kwargs = solver(result, kwargs)
        return result
    def fit(self, metadata: Metadata, dataloader: DataLoader):
        def fit_one_processor(d: DataProcessor, metadata: Metadata, dataloader: DataLoader):
            if dataloader:
                d.fit(metadata=metadata, tabular_data=dataloader)
            else:
                d.fit(metadata=metadata)
        def fit_result_solver(result, kwargs):
            return kwargs
        self._parallel_wrapper(
            ddlist=self.data_processors_execute_order_list,
            desc="Preparing Data",
            func=fit_one_processor,
            kwargs={"metadata": metadata, "dataloader": dataloader},
            solver=fit_result_solver
        )
        return self

    def convert(self, data: DataFrame):
        def convert_one_processor(d: DataProcessor, data: DataFrame):
            return d.convert(data)
        def convert_result_solver(result, kwargs):
            kwargs["data"] = result
            return kwargs
        return self._parallel_wrapper(
            ddlist=self.data_processors_execute_order_list,
            desc="Transforming Data",
            func=convert_one_processor,
            kwargs={"data": data.copy()},
            solver=convert_result_solver
        )

    def reverse_convert(self, data: DataFrame):
        def convert_one_processor(d: DataProcessor, data: DataFrame):
            return d.reverse_convert(data)
        def convert_result_solver(result, kwargs):
            kwargs["data"] = result
            return kwargs
        return self._parallel_wrapper(
            ddlist=list(reversed(self.data_processors_execute_order_list)),
            desc="Reverse Transforming",
            func=convert_one_processor,
            kwargs={"data": data},
            solver=convert_result_solver
        )

