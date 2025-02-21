import { Alert, Form, FormInstance, Input, InputNumber, ModalFuncProps, Select } from 'antd';
import { number, string, undefined as undefinedType, union } from 'io-ts';
import yaml from 'js-yaml';
import React, { useCallback, useEffect, useMemo, useState } from 'react';

import Button from 'components/kit/Button';
import Link from 'components/Link';
import { SettingsConfig, useSettings } from 'hooks/useSettings';
import { getTaskTemplates } from 'services/api';
import Spinner from 'shared/components/Spinner/Spinner';
import useModal, { ModalHooks } from 'shared/hooks/useModal/useModal';
import usePrevious from 'shared/hooks/usePrevious';
import { RawJson } from 'shared/types';
import { useResourcePools } from 'stores/resourcePools';
import { Template } from 'types';
import handleError from 'utils/error';
import { JupyterLabOptions, launchJupyterLab, previewJupyterLab } from 'utils/jupyter';
import { Loadable } from 'utils/loadable';

import css from './useModalJupyterLab.module.scss';

const { Option } = Select;

const STORAGE_PATH = 'jupyter-lab';
const DEFAULT_SLOT_COUNT = 1;

const settingsConfig: SettingsConfig<JupyterLabOptions> = {
  settings: {
    name: {
      defaultValue: '',
      skipUrlEncoding: true,
      storageKey: 'name',
      type: union([string, undefinedType]),
    },
    pool: {
      defaultValue: '',
      skipUrlEncoding: true,
      storageKey: 'pool',
      type: union([string, undefinedType]),
    },
    slots: {
      defaultValue: DEFAULT_SLOT_COUNT,
      skipUrlEncoding: true,
      storageKey: 'slots',
      type: union([number, undefinedType]),
    },
    template: {
      defaultValue: undefined,
      skipUrlEncoding: true,
      storageKey: 'template',
      type: union([string, undefinedType]),
    },
  },
  storagePath: STORAGE_PATH,
};

interface FullConfigProps {
  config?: string;
  configError?: string;
  onChange?: (config: string) => void;
  setButtonDisabled: (buttonDisabled: boolean) => void;
}

const MonacoEditor = React.lazy(() => import('components/MonacoEditor'));

const useModalJupyterLab = (): ModalHooks => {
  const [visible, setVisible] = useState(false);
  const [showFullConfig, setShowFullConfig] = useState(false);
  const [config, setConfig] = useState<string>();
  const [configError, setConfigError] = useState<string>();
  const [buttonDisabled, setButtonDisabled] = useState(false);
  const previousConfig = usePrevious(config, config);
  const previousShowConfig = usePrevious(showFullConfig, showFullConfig);
  const [form] = Form.useForm<JupyterLabOptions>();

  const { settings: defaults, updateSettings: updateDefaults } =
    useSettings<JupyterLabOptions>(settingsConfig);

  const handleModalClose = useCallback(() => {
    setVisible(false);
    const fields: JupyterLabOptions = form.getFieldsValue(true);
    updateDefaults(fields);
  }, [form, updateDefaults]);

  const {
    modalClose,
    modalOpen: openOrUpdate,
    ...modalHook
  } = useModal({ onClose: handleModalClose });

  const fetchConfig = useCallback(async () => {
    const fields: JupyterLabOptions = form.getFieldsValue(true);
    try {
      const newConfig = await previewJupyterLab({
        name: fields?.name,
        pool: fields?.pool,
        slots: fields?.slots,
        template: fields?.template,
      });
      setConfig(yaml.dump(newConfig));
    } catch (e) {
      setConfig(undefined);
      setConfigError('Unable to fetch JupyterLab config.');
    }
  }, [form]);

  const handleSecondary = useCallback(() => {
    if (showFullConfig) setButtonDisabled(false);
    setShowFullConfig((show) => !show);
  }, [showFullConfig]);

  const handleSubmit = useCallback(() => {
    const fields: JupyterLabOptions = form.getFieldsValue(true);
    updateDefaults(fields);
    if (showFullConfig) {
      launchJupyterLab({ config: yaml.load(config || '') as RawJson });
    } else {
      launchJupyterLab({
        name: fields?.name,
        pool: fields?.pool,
        slots: fields?.slots,
        template: fields?.template,
      });
    }
    modalClose();
    setVisible(false);
  }, [config, form, showFullConfig, modalClose, updateDefaults]);

  const handleConfigChange = useCallback((config: string) => {
    setConfig(config);
    setConfigError(undefined);
  }, []);

  const bodyContent = useMemo(() => {
    return showFullConfig ? (
      <JupyterLabFullConfig
        config={config}
        configError={configError}
        setButtonDisabled={setButtonDisabled}
        onChange={handleConfigChange}
      />
    ) : (
      <JupyterLabForm defaults={defaults} form={form} />
    );
  }, [config, configError, handleConfigChange, showFullConfig, defaults, form]);

  const content = useMemo(
    () => (
      <>
        {bodyContent}
        <div className={css.buttons}>
          <Button onClick={handleSecondary}>
            {showFullConfig ? 'Show Simple Config' : 'Show Full Config'}
          </Button>
          <Button disabled={buttonDisabled} type="primary" onClick={handleSubmit}>
            Launch
          </Button>
        </div>
      </>
    ),
    [bodyContent, buttonDisabled, handleSubmit, handleSecondary, showFullConfig],
  );

  const modalProps: ModalFuncProps = useMemo(
    () => ({
      className: css.noFooter,
      closable: true,
      content,
      icon: null,
      title: 'Launch JupyterLab',
      width: showFullConfig ? 1000 : undefined,
    }),
    [content, showFullConfig],
  );

  const modalOpen = useCallback(
    (initialModalProps: ModalFuncProps = {}) => {
      setVisible(true);
      openOrUpdate({ ...modalProps, ...initialModalProps });
    },
    [modalProps, openOrUpdate],
  );

  // Fetch full config when showing advanced mode.
  useEffect(() => {
    if (showFullConfig) {
      fetchConfig();
    }
  }, [fetchConfig, showFullConfig]);

  // Update the modal when user toggles the `Show Full Config` button.
  useEffect(() => {
    if (visible && (config !== previousConfig || showFullConfig !== previousShowConfig)) {
      openOrUpdate(modalProps);
    }
  }, [
    config,
    modalProps,
    openOrUpdate,
    previousConfig,
    previousShowConfig,
    showFullConfig,
    visible,
  ]);

  return { modalClose, modalOpen, ...modalHook };
};

const JupyterLabFullConfig: React.FC<FullConfigProps> = ({
  config,
  configError,
  onChange,
  setButtonDisabled,
}: FullConfigProps) => {
  const [field, setField] = useState([{ name: 'config', value: '' }]);

  const handleConfigChange = useCallback(
    (_: unknown, allFields: unknown) => {
      if (!Array.isArray(allFields) || allFields.length === 0) return;
      try {
        const configString = allFields[0].value;
        onChange?.(configString);
      } catch (e) {
        handleError(e);
      }
    },
    [onChange],
  );

  useEffect(() => {
    setField([{ name: 'config', value: config || '' }]);
  }, [config]);

  return (
    <Form fields={field} onFieldsChange={handleConfigChange}>
      <div className={css.note}>
        <Link external path="/docs/reference/api/command-notebook-config.html">
          Read about JupyterLab settings
        </Link>
      </div>
      <React.Suspense
        fallback={
          <div className={css.loading}>
            <Spinner tip="Loading text editor..." />
          </div>
        }>
        <Form.Item
          name="config"
          rules={[
            { message: 'JupyterLab config required', required: true },
            {
              validator: (rule, value) => {
                try {
                  yaml.load(value);
                  setButtonDisabled(false);
                  return Promise.resolve();
                } catch (err: unknown) {
                  setButtonDisabled(true);
                  return Promise.reject(
                    new Error(
                      `Invalid YAML on line ${(err as { mark: { line: string } }).mark.line}.`,
                    ),
                  );
                }
              },
            },
          ]}>
          <MonacoEditor
            height="40vh"
            options={{
              wordWrap: 'on',
              wrappingIndent: 'indent',
            }}
          />
        </Form.Item>
      </React.Suspense>
      {configError && <Alert message={configError} type="error" />}
    </Form>
  );
};

const JupyterLabForm: React.FC<{
  defaults: JupyterLabOptions;
  form: FormInstance<JupyterLabOptions>;
}> = ({ form, defaults }) => {
  const [templates, setTemplates] = useState<Template[]>([]);

  const loadableResourcePools = useResourcePools();
  const resourcePools = Loadable.getOrElse([], loadableResourcePools); // TODO show spinner when this is loading

  const selectedPoolName = Form.useWatch('pool', form);

  const resourceInfo = useMemo(() => {
    const selectedPool = resourcePools.find((pool) => pool.name === selectedPoolName);
    if (!selectedPool) return { hasAux: false, hasCompute: false, maxSlots: 0 };

    /**
     * For static resource pools, the slots-per-agent comes through as -1,
     * meaning it is unknown how many we may have.
     */
    const hasAuxCapacity = selectedPool.auxContainerCapacityPerAgent > 0;
    const hasSlots = selectedPool.slotsAvailable > 0;
    const maxSlots = selectedPool.slotsPerAgent ?? 0;
    const hasSlotsPerAgent = maxSlots !== 0;
    const hasComputeCapacity = hasSlots || hasSlotsPerAgent;

    return {
      hasAux: hasAuxCapacity,
      hasCompute: hasComputeCapacity,
      maxSlots: maxSlots,
    };
  }, [selectedPoolName, resourcePools]);

  useEffect(() => {
    if (!resourceInfo.hasCompute && resourceInfo.hasAux) form.setFieldValue('slots', 0);
    else if (resourceInfo.hasCompute) {
      const slots = form.getFieldValue('slots');
      if (slots == null) form.setFieldValue('slots', DEFAULT_SLOT_COUNT);
    }
  }, [resourceInfo, form]);

  const fetchTemplates = useCallback(async () => {
    try {
      setTemplates(await getTaskTemplates({}));
    } catch (e) {
      handleError(e);
    }
  }, []);

  useEffect(() => {
    fetchTemplates();
  }, [fetchTemplates]);

  useEffect(() => {
    const fields = form.getFieldsValue(true);
    if (!fields?.pool && resourcePools[0]?.name) {
      const firstPoolInList = resourcePools[0]?.name;
      form.setFieldValue('pool', firstPoolInList);
    }
  }, [resourcePools, form]);

  return (
    <Form className={css.form} form={form} initialValues={defaults}>
      <Form.Item className={css.line} label="Template" name="template">
        <Select allowClear placeholder="No template (optional)">
          {templates.map((temp) => (
            <Option key={temp.name} value={temp.name}>
              {temp.name}
            </Option>
          ))}
        </Select>
      </Form.Item>
      <Form.Item className={css.line} label="Name" name="name">
        <Input placeholder="Name (optional)" />
      </Form.Item>
      <Form.Item className={css.line} label="Resource Pool" name="pool">
        <Select allowClear placeholder="Pick the best option">
          {resourcePools.map((pool) => (
            <Option key={pool.name} value={pool.name}>
              {pool.name}
            </Option>
          ))}
        </Select>
      </Form.Item>
      <Form.Item className={css.line} hidden={!resourceInfo.hasCompute} label="Slots" name="slots">
        <InputNumber
          max={resourceInfo.maxSlots === -1 ? Number.MAX_SAFE_INTEGER : resourceInfo.maxSlots}
          min={0}
        />
      </Form.Item>
    </Form>
  );
};

export default useModalJupyterLab;
