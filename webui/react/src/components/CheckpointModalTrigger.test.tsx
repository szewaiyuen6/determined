import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React, { useEffect } from 'react';

import CheckpointModalTrigger from 'components/CheckpointModalTrigger';
import { StoreProvider as UIProvider } from 'shared/contexts/stores/UI';
import { AuthProvider, useAuth } from 'stores/auth';
import { generateTestExperimentData } from 'storybook/shared/generateTestData';

const TEST_MODAL_TITLE = 'Checkpoint Modal Test';
const REGISTER_CHECKPOINT_TEXT = 'Register Checkpoint';

jest.mock('services/api', () => ({
  getModels: () => {
    return Promise.resolve({ models: [] });
  },
}));

const user = userEvent.setup();

const ModalTrigger: React.FC = () => {
  const { experiment, checkpoint } = generateTestExperimentData();

  const { setAuth } = useAuth();

  useEffect(() => {
    setAuth({ isAuthenticated: true });
  }, [setAuth]);

  return (
    <CheckpointModalTrigger
      checkpoint={checkpoint}
      experiment={experiment}
      title={TEST_MODAL_TITLE}
    />
  );
};

const setup = async () => {
  render(
    <AuthProvider>
      <UIProvider>
        <ModalTrigger />
      </UIProvider>
    </AuthProvider>,
  );

  await user.click(screen.getByRole('button'));

  return user;
};

describe('CheckpointModalTrigger', () => {
  it('open modal', async () => {
    await setup();

    expect(await screen.findByText(TEST_MODAL_TITLE)).toBeInTheDocument();
  });

  it('close modal', async () => {
    const user = await setup();

    await screen.findByText(TEST_MODAL_TITLE);

    await user.click(screen.getByRole('button', { name: /cancel/i }));

    await waitFor(() => {
      expect(screen.queryByText(TEST_MODAL_TITLE)).not.toBeInTheDocument();
    });
  });

  it('open register checkpoint modal', async () => {
    const user = await setup();

    await screen.findByText(TEST_MODAL_TITLE);

    await user.click(screen.getByRole('button', { name: REGISTER_CHECKPOINT_TEXT }));

    await waitFor(() => {
      // One for the title and one for the confirm button label.
      const elements = screen.queryAllByText(REGISTER_CHECKPOINT_TEXT);
      expect(elements).toHaveLength(2);
    });
  });
});
