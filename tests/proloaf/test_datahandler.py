import pytest
import numpy as np
import pandas as pd
from proloaf.datahandler import MultiScaler, scale_all
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


@pytest.fixture
def dataframe():
    np.random.seed(1)
    cols = ["target", "feat1", "feat2", "feat3"]
    X = pd.DataFrame(np.random.randn(10, len(cols)), columns=cols)
    return X


@pytest.fixture
def feature_groups_with_name():
    return [
        {"name": "grp_mm", "scaler": ["minmax", -1.0, 1.0], "features": ["feat1"]},
        {"name": "grp_rob", "scaler": ["robust", 0.25, 0.75], "features": ["feat2"]},
        {"name": "grp_std", "scaler": ["standard"], "features": ["feat3"]},
    ]


@pytest.fixture
def feature_groups():
    return [
        {"scaler": ["minmax", -1.0, 1.0], "features": ["feat1"]},
        {"scaler": ["robust", 0.25, 0.75], "features": ["feat2"]},
        {"name": "grp_std", "scaler": ["standard"], "features": ["feat3"]},
    ]


@pytest.fixture
def multi_scaler_fit(feature_groups, dataframe):
    return MultiScaler(feature_groups).fit(dataframe)


class TestMultiScaler:
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_same_as_scale_all(self, feature_groups_with_name, dataframe):
        scaler = MultiScaler(feature_groups=feature_groups_with_name)
        scaler_out = scaler.fit_transform(dataframe)
        scale_all_out = scale_all(dataframe, feature_groups_with_name)
        # not specified features are droped in scale_all and not in MultiScaler
        # this is a deliberate change
        # scale_all also returns the scalers that are now contained in MultiScaler
        # also deliberate
        assert (scaler_out[["feat1", "feat2", "feat3"]] == scale_all_out[0]).all(None)

    def test_init(self, feature_groups):
        scaler = MultiScaler(feature_groups)
        assert scaler

    def test_fit(self, feature_groups, dataframe):
        scaler = MultiScaler(feature_groups)
        scaler.fit(dataframe)
        for sc in scaler.scalers.values():
            check_is_fitted(sc)
        check_is_fitted(scaler)

    def test_fit_prefit_scalers(self, feature_groups, dataframe):
        aux_scaler = MultiScaler(feature_groups)
        aux_scaler.fit(dataframe)
        scaler = MultiScaler(feature_groups, scalers=aux_scaler.scalers)
        for sc in scaler.scalers.values():
            check_is_fitted(sc)
        check_is_fitted(scaler)

    def test_fit_prefit_scalers(self, feature_groups, dataframe):
        aux_scaler = MultiScaler(feature_groups)
        aux_scaler.fit(dataframe)
        scaler = MultiScaler(
            feature_groups, scalers={"feat1": aux_scaler.scalers["feat1"]}
        )
        with pytest.raises(NotFittedError):
            check_is_fitted(scaler)

    def test_not_fit(self, feature_groups, dataframe):
        scaler = MultiScaler(feature_groups)
        with pytest.raises(NotFittedError):
            check_is_fitted(scaler)

    def test_manual_transform(self, multi_scaler_fit, dataframe):
        direct_result = multi_scaler_fit.transform(dataframe)[["feat2"]]
        indirect_result = multi_scaler_fit.manual_transform(
            dataframe[["feat2"]], "feat2"
        )
        print(direct_result)
        print(indirect_result)
        assert (direct_result == indirect_result).all(None)

    def test_manual_inverse_transform(self, multi_scaler_fit, dataframe):
        direct_result = multi_scaler_fit.inverse_transform(dataframe)[["feat2"]]
        indirect_result = multi_scaler_fit.manual_inverse_transform(
            dataframe[["feat2"]], "feat2"
        )
        print(direct_result)
        print(indirect_result)
        assert (direct_result == indirect_result).all(None)